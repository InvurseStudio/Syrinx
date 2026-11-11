//! A small HTTP/1.1 server side: one request per connection, read whole (the app's
//! requests are small), and replies written whole or streamed (server-sent events).

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;

pub struct Request {
    pub method: String,
    /// The path without its query.
    pub path: String,
    pub query: String,
    pub bearer: Option<String>,
    pub body: Vec<u8>,
}

/// The largest body read (uploads aren't served, so requests stay small).
const MAX_BODY: usize = 32 << 20;

pub fn read_request(stream: &TcpStream) -> Result<Request, String> {
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line).map_err(|e| e.to_string())?;
    let mut parts = line.split_whitespace();
    let method = parts.next().ok_or("empty request")?.to_string();
    let target = parts.next().ok_or("no path")?.to_string();
    let (path, query) = match target.split_once('?') {
        Some((p, q)) => (p.to_string(), q.to_string()),
        None => (target, String::new()),
    };
    let mut length = 0usize;
    let mut chunked = false;
    let mut bearer = None;
    loop {
        let mut header = String::new();
        if reader.read_line(&mut header).map_err(|e| e.to_string())? == 0 {
            break;
        }
        let header = header.trim_end();
        if header.is_empty() {
            break;
        }
        let Some((name, value)) = header.split_once(':') else {
            continue;
        };
        let value = value.trim();
        if name.eq_ignore_ascii_case("content-length") {
            length = value.parse().map_err(|_| "bad Content-Length")?;
        } else if name.eq_ignore_ascii_case("transfer-encoding") {
            // .NET's PostAsJsonAsync sends its body in chunks, with no length up front.
            chunked = value.to_ascii_lowercase().contains("chunked");
        } else if name.eq_ignore_ascii_case("authorization") {
            bearer = value.strip_prefix("Bearer ").map(|t| t.trim().to_string());
        }
    }
    let body = if chunked {
        read_chunks(&mut reader)?
    } else {
        if length > MAX_BODY {
            return Err("request too large".into());
        }
        let mut body = vec![0; length];
        reader.read_exact(&mut body).map_err(|e| e.to_string())?;
        body
    };
    Ok(Request {
        method,
        path,
        query,
        bearer,
        body,
    })
}

/// A chunked body: hex sizes, each chunk, a zero-sized last one, then any trailers.
fn read_chunks(reader: &mut impl BufRead) -> Result<Vec<u8>, String> {
    let mut body = Vec::new();
    loop {
        let mut size = String::new();
        reader.read_line(&mut size).map_err(|e| e.to_string())?;
        let hex = size.trim().split(';').next().unwrap_or("");
        let n = usize::from_str_radix(hex, 16).map_err(|_| "bad chunk size")?;
        if n == 0 {
            // Trailers, up to a blank line.
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).map_err(|e| e.to_string())? == 0
                    || line.trim().is_empty()
                {
                    return Ok(body);
                }
            }
        }
        if body.len() + n > MAX_BODY {
            return Err("request too large".into());
        }
        let start = body.len();
        body.resize(start + n, 0);
        reader
            .read_exact(&mut body[start..])
            .map_err(|e| e.to_string())?;
        let mut end = [0u8; 2];
        reader.read_exact(&mut end).map_err(|e| e.to_string())?; // the chunk's CRLF
    }
}

pub fn respond(stream: &mut TcpStream, status: u16, json: &str) {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        409 => "Conflict",
        _ => "Error",
    };
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        json.len()
    );
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(json.as_bytes());
    let _ = stream.flush();
}

/// Starts a text/event-stream reply; then write events with [`event`].
pub fn start_events(stream: &mut TcpStream) -> std::io::Result<()> {
    stream.write_all(
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n",
    )?;
    stream.flush()
}

/// One "data:" event. Fails once the client has gone.
pub fn event(stream: &mut TcpStream, data: &str) -> std::io::Result<()> {
    stream.write_all(format!("data: {data}\n\n").as_bytes())?;
    stream.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_chunked_bodies() {
        let mut input: &[u8] = b"5\r\nhello\r\n7;x=1\r\n, world\r\n0\r\n\r\n";
        assert_eq!(read_chunks(&mut input).unwrap(), b"hello, world");
    }
}
