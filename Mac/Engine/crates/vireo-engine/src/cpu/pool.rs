//! A thread pool for splitting one piece of work, such as a matrix-vector product,
//! across every core.
//!
//! The threads are started once and wait for work, because starting threads costs far
//! more than the work a single layer of a small model gives each of them. The calling
//! thread works too, and `run` returns only when every task has finished, so tasks can
//! borrow from the caller's stack.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

type Task = dyn Fn(usize) + Sync;

struct State {
    /// The current job. Its real lifetime is `run`'s, which outlasts every use of it.
    job: Option<&'static Task>,
    tasks: usize,
    /// Counts jobs, so each worker takes part in each job exactly once.
    epoch: u64,
    /// Workers still on the current job.
    busy: usize,
    shutdown: bool,
}

struct Shared {
    state: Mutex<State>,
    start: Condvar,
    finished: Condvar,
    next_task: AtomicUsize,
}

pub struct Pool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
}

impl Pool {
    /// A pool that runs work on `threads` threads in all, the caller's included.
    pub fn new(threads: usize) -> Self {
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                job: None,
                tasks: 0,
                epoch: 0,
                busy: 0,
                shutdown: false,
            }),
            start: Condvar::new(),
            finished: Condvar::new(),
            next_task: AtomicUsize::new(0),
        });
        let workers = (1..threads.max(1))
            .map(|i| {
                let shared = Arc::clone(&shared);
                std::thread::Builder::new()
                    .name(format!("vireo-cpu-{i}"))
                    .spawn(move || worker(&shared))
                    .expect("couldn't start a worker thread")
            })
            .collect();
        Self { shared, workers }
    }

    /// One thread per logical core.
    pub fn with_all_cores() -> Self {
        Self::new(std::thread::available_parallelism().map_or(1, |n| n.get()))
    }

    pub fn threads(&self) -> usize {
        self.workers.len() + 1
    }

    /// Runs `task(0)`, `task(1)`, …, `task(tasks - 1)` across the pool and waits for all
    /// of them. Tasks run in no particular order, and must not panic.
    pub fn run(&self, tasks: usize, task: &(dyn Fn(usize) + Sync)) {
        if tasks <= 1 || self.workers.is_empty() {
            (0..tasks).for_each(task);
            return;
        }
        // Safety: the reference is only used by workers during this call. `run` doesn't
        // return until `busy` is 0, and workers stop using the job before lowering it.
        let job: &'static Task =
            unsafe { std::mem::transmute::<&(dyn Fn(usize) + Sync + '_), &'static Task>(task) };
        {
            let mut s = self.shared.state.lock().unwrap();
            s.job = Some(job);
            s.tasks = tasks;
            s.epoch += 1;
            s.busy = self.workers.len();
            self.shared.next_task.store(0, Ordering::Relaxed);
        }
        self.shared.start.notify_all();
        take_tasks(&self.shared, task, tasks);
        let mut s = self.shared.state.lock().unwrap();
        while s.busy > 0 {
            s = self.shared.finished.wait(s).unwrap();
        }
        s.job = None;
    }
}

fn take_tasks(shared: &Shared, task: &(dyn Fn(usize) + Sync), tasks: usize) {
    loop {
        let i = shared.next_task.fetch_add(1, Ordering::Relaxed);
        if i >= tasks {
            return;
        }
        task(i);
    }
}

fn worker(shared: &Shared) {
    let mut seen = 0;
    loop {
        let (job, tasks) = {
            let mut s = shared.state.lock().unwrap();
            while s.epoch == seen && !s.shutdown {
                s = shared.start.wait(s).unwrap();
            }
            if s.shutdown {
                return;
            }
            seen = s.epoch;
            (s.job.expect("a job comes with each new epoch"), s.tasks)
        };
        take_tasks(shared, job, tasks);
        let mut s = shared.state.lock().unwrap();
        s.busy -= 1;
        if s.busy == 0 {
            shared.finished.notify_one();
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.state.lock().unwrap().shutdown = true;
        self.shared.start.notify_all();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    #[test]
    fn runs_every_task_once() {
        let pool = Pool::new(8);
        for round in 0..200 {
            let tasks = 1 + round % 37;
            let hits: Vec<AtomicU64> = (0..tasks).map(|_| AtomicU64::new(0)).collect();
            pool.run(tasks, &|i| {
                hits[i].fetch_add(1, Ordering::Relaxed);
            });
            assert!(
                hits.iter().all(|h| h.load(Ordering::Relaxed) == 1),
                "round {round}"
            );
        }
    }

    #[test]
    fn borrows_from_the_caller() {
        let pool = Pool::new(4);
        let input: Vec<u64> = (0..1000).collect();
        let total = AtomicU64::new(0);
        pool.run(10, &|i| {
            let part: u64 = input[i * 100..(i + 1) * 100].iter().sum();
            total.fetch_add(part, Ordering::Relaxed);
        });
        assert_eq!(total.into_inner(), 999 * 1000 / 2);
    }

    #[test]
    fn a_single_thread_pool_works() {
        let pool = Pool::new(1);
        let count = AtomicU64::new(0);
        pool.run(5, &|_| {
            count.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(count.into_inner(), 5);
    }
}
