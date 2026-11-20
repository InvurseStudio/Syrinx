// The Connect page: what another chat app needs to use the model. Syrinx speaks the OpenAI
// chat API, so any app that lets you add an OpenAI-compatible server works: the base URL
// (on this Mac, and on the network when that's on), the API key, and the model's name.
import AppKit
import SwiftUI

struct ConnectPage: View {
    @EnvironmentObject private var settings: Settings
    @EnvironmentObject private var server: ServerProcess

    private var port: Int { server.address?.port ?? settings.port }
    private var localBase: String { "http://127.0.0.1:\(port)/v1" }
    private var key: String { settings.apiKey.trimmingCharacters(in: .whitespaces) }
    private var modelName: String {
        (server.model ?? settings.model).map { (($0 as NSString).lastPathComponent as NSString).deletingPathExtension } ?? "local"
    }

    var body: some View {
        Page(title: "Connect") {
            Caption("Syrinx speaks the OpenAI chat API. In any app that lets you add an OpenAI-compatible server or provider, use the details below.")
            if server.state != .running {
                Banner(tone: .caution, title: "The server isn't running",
                       message: "Start it on the Server page. These are the details it will use.")
            }
            SettingsSection("Base URL") {
                SettingRow("On this Mac", "For apps running on this computer.") { CopyField(value: localBase) }
                if settings.onNetwork {
                    ForEach(NetworkAddresses.current(), id: \.self) { ip in
                        SettingRow("On the network (\(ip))", "For phones and other computers on the same network.") {
                            CopyField(value: "http://\(ip):\(port)/v1")
                        }
                    }
                } else {
                    Caption("Only apps on this Mac can connect. To let other devices in, turn on \"Let other devices on this network connect\" on the Server page.")
                }
            }
            SettingsSection("Sign-in") {
                SettingRow("API key", key.isEmpty
                           ? "No key is set: apps on this Mac can give anything (some apps need a value; any will do)."
                           : "Give this as the app's API key.") {
                    CopyField(value: key.isEmpty ? "none" : key)
                }
                SettingRow("Model", "Syrinx serves one model; apps that ask for a name can use this one, or any.") {
                    CopyField(value: modelName)
                }
            }
            SettingsSection("In a browser") {
                SettingRow("Syrinx's own chat page",
                           "A Vireo-style chat in any browser on this Mac" + (settings.onNetwork ? ", or on the network at http://<address>:\(port)/" : "") + ".") {
                    Button {
                        if let url = URL(string: "http://127.0.0.1:\(port)/") { NSWorkspace.shared.open(url) }
                    } label: { Label("Open the chat page", systemImage: "bubble.left.and.bubble.right") }
                    .disabled(server.state != .running)
                }
            }
            VStack(alignment: .leading, spacing: 10) {
                SectionHeader("From code")
                Example(title: "Python (the openai package)", code: python)
                Example(title: "curl", code: curl)
            }
        }
    }

    private var python: String {
        """
        from openai import OpenAI

        client = OpenAI(base_url="\(localBase)", api_key="\(key.isEmpty ? "none" : key)")
        reply = client.chat.completions.create(
            model="\(modelName)",
            messages=[{"role": "user", "content": "Hello!"}],
        )
        print(reply.choices[0].message.content)
        """
    }

    private var curl: String {
        var lines = ["curl \(localBase)/chat/completions \\", "  -H \"Content-Type: application/json\" \\"]
        if !key.isEmpty { lines.append("  -H \"Authorization: Bearer \(key)\" \\") }
        lines.append("  -d '{\"model\": \"\(modelName)\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}]}'")
        return lines.joined(separator: "\n")
    }
}

private struct Example: View {
    let title: String
    let code: String

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(title).font(.system(size: 13))
                Spacer()
                CopyButton(value: { code }, help: "Copy the example")
            }
            ScrollView(.horizontal) {
                Text(code)
                    .font(.system(size: 12, design: .monospaced))
                    .textSelection(.enabled)
                    .fixedSize()
                    .padding(EdgeInsets(top: 10, leading: 12, bottom: 10, trailing: 12))
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Theme.bgTertiary, in: RoundedRectangle(cornerRadius: 6))
        }
    }
}

/// This Mac's IPv4 addresses on networks that are up (not loopback, tunnels or self-assigned).
enum NetworkAddresses {
    static func current() -> [String] {
        var found: [String] = []
        var list: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&list) == 0, let first = list else { return [] }
        defer { freeifaddrs(list) }
        for pointer in sequence(first: first, next: { $0.pointee.ifa_next }) {
            let entry = pointer.pointee
            let flags = Int32(entry.ifa_flags)
            guard let address = entry.ifa_addr, address.pointee.sa_family == UInt8(AF_INET),
                  flags & IFF_UP != 0, flags & IFF_RUNNING != 0, flags & IFF_LOOPBACK == 0 else { continue }
            let name = String(cString: entry.ifa_name)
            if name.hasPrefix("utun") || name.hasPrefix("ipsec") || name.hasPrefix("ppp") { continue }
            var host = [CChar](repeating: 0, count: Int(NI_MAXHOST))
            guard getnameinfo(address, socklen_t(address.pointee.sa_len), &host, socklen_t(host.count), nil, 0, NI_NUMERICHOST) == 0 else { continue }
            let ip = String(cString: host)
            if !ip.hasPrefix("169.254."), !found.contains(ip) { found.append(ip) }
        }
        return found
    }
}
