//! Syrinx on its own (`--standalone`), apart from Vireo: the OpenAI-compatible API
//! (`openai.rs`, always on here), a chat page for the browser, the statistics, and nothing
//! of the Vireo app's (no pairing, tools, tasks or uploads). Its chats' caches and the rest
//! are kept in Syrinx's own folder (`store::use_own_folder`).
//!
//!   GET  /            the chat page (`standalone.html`)
//!   GET  /health      {status, model, runtime, context, key}: whether it's up, and whether
//!                     callers need a key
//!   GET  /stats       the statistics dashboard (`stats.html`), as the Vireo apps show it
//!   GET  /api/stats   its figures (`stats.rs`), behind the API's keys
//!   /v1/...           the OpenAI-compatible API (`openai.rs`), behind the API's keys
//!
//! Who may call: anyone on this computer when no key was given; with `--api-key`, those
//! who give it; and the app that started the server, with its `--token` (the Syrinx
//! manager, for the statistics). `developer.rs` checks them all.

use std::net::TcpStream;

use crate::http::{self, Request};
use crate::json;
use crate::{Shared, developer, openai, stats};

/// The chat page, at `/`.
const CHAT_PAGE: &str = include_str!("standalone.html");

/// The statistics dashboard, at `/stats`. It reads `/api/stats` with the key, or the
/// manager's token, given after `#token=` in its address, or asks for the key.
const STATS_PAGE: &str = include_str!("stats.html");

/// The runtime as Syrinx names itself on its own: "Syrinx (GPU, 8-bit)".
pub fn runtime(shared: &Shared) -> String {
    shared.info.runtime.replacen("Vireo engine", "Syrinx", 1)
}

/// Every request in standalone mode.
pub fn serve(mut stream: TcpStream, request: Request, shared: &Shared) {
    let path = request.path.trim_end_matches('/').to_string();
    let method = request.method.as_str();
    // A browser's check before a page on another site calls (`http::allow_cross_origin`).
    if method == "OPTIONS" {
        return http::respond_with(&mut stream, 200, "text/plain", b"");
    }
    if path == "/v1" || path.starts_with("/v1/") {
        return openai::serve(stream, shared, &request, &path);
    }
    match (method, path.as_str()) {
        ("GET", "" | "/index.html") => {
            http::respond_with(&mut stream, 200, "text/html; charset=utf-8", CHAT_PAGE.as_bytes())
        }
        ("GET", "/stats") => {
            http::respond_with(&mut stream, 200, "text/html; charset=utf-8", STATS_PAGE.as_bytes())
        }
        ("GET", "/health") => {
            let i = &shared.info;
            http::respond(
                &mut stream,
                200,
                &format!(
                    "{{\"status\":\"ok\",\"model\":{},\"runtime\":{},\"context\":{},\"key\":{}}}",
                    json::string(&i.model),
                    json::string(&runtime(shared)),
                    i.context,
                    developer::command_line_key()
                ),
            )
        }
        ("GET", "/api/stats") => {
            if let Err((status, code, why)) = developer::authorize(&developer::load(), request.bearer.as_deref()) {
                stats::refuse(&crate::peer(&stream), "Statistics refused");
                return http::respond(
                    &mut stream,
                    status,
                    &openai::error_json(&why, "authentication_error", code),
                );
            }
            let i = &shared.info;
            http::respond(&mut stream, 200, &stats::json(&i.model, &runtime(shared), i.context))
        }
        _ => http::respond(
            &mut stream,
            404,
            &openai::error_json("Not on this server.", "not_found_error", "unknown_url"),
        ),
    }
}
