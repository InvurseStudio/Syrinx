// The Statistics page: the server's own dashboard (/stats, stats.html in syrinx-server: the
// Vireo Mac app's Statistics window as a page), shown here with the token this app started
// the server with, so it opens even when the server has an API key.
import AppKit
import SwiftUI
import WebKit

struct StatsPage: View {
    @EnvironmentObject private var server: ServerProcess

    private var url: URL? {
        guard server.state == .running, let port = server.address?.port else { return nil }
        return URL(string: "http://127.0.0.1:\(port)/stats#token=\(server.token)")
    }

    var body: some View {
        if let url {
            StatsWebView(url: url)
                .overlay(alignment: .topTrailing) {
                    Button {
                        NSWorkspace.shared.open(url)
                    } label: { Label("Open in browser", systemImage: "safari") }
                    .padding(12)
                }
        } else {
            VStack(spacing: 10) {
                Image(systemName: "chart.bar").font(.system(size: 40)).foregroundStyle(Theme.textSecondary)
                Text("The server isn't running").font(.system(size: 17, weight: .semibold))
                Caption("Start it to see how the model is being used: requests, speed, the prompt cache, and which apps are connected.")
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: 420)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }
}

private struct StatsWebView: NSViewRepresentable {
    let url: URL

    func makeNSView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()
        // Nothing kept between launches: the token changes each time.
        configuration.websiteDataStore = .nonPersistent()
        let view = WKWebView(frame: .zero, configuration: configuration)
        // The page's background before it draws, not white.
        view.setValue(false, forKey: "drawsBackground")
        view.underPageBackgroundColor = Theme.backgroundNS
        view.navigationDelegate = context.coordinator
        view.load(URLRequest(url: url))
        context.coordinator.shown = url
        return view
    }

    func updateNSView(_ view: WKWebView, context: Context) {
        guard context.coordinator.shown != url else { return }
        context.coordinator.shown = url
        view.load(URLRequest(url: url))
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    final class Coordinator: NSObject, WKNavigationDelegate {
        var shown: URL?

        /// Links out of the dashboard open in the browser.
        func webView(_ webView: WKWebView, decidePolicyFor action: WKNavigationAction,
                     decisionHandler: @escaping @MainActor (WKNavigationActionPolicy) -> Void) {
            if action.navigationType == .linkActivated, let target = action.request.url {
                NSWorkspace.shared.open(target)
                decisionHandler(.cancel)
                return
            }
            decisionHandler(.allow)
        }
    }
}
