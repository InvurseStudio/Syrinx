// The window: a sidebar of pages with the server's state at its foot, and the page beside
// it. Each page is made once and kept, so the statistics keep their place.
import SwiftUI

struct MainView: View {
    @EnvironmentObject private var app: AppModel
    @EnvironmentObject private var server: ServerProcess

    var body: some View {
        HStack(spacing: 0) {
            sidebar
                .frame(width: 240)
                .background(Theme.bgSecondary)
                .overlay(alignment: .trailing) { Rectangle().fill(Theme.border).frame(width: 1) }
            ZStack {
                ForEach(AppModel.PageName.allCases) { name in
                    page(name)
                        .opacity(app.page == name ? 1 : 0)
                        .allowsHitTesting(app.page == name)
                        .accessibilityHidden(app.page != name)
                }
            }
            .padding(.top, 28)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .background(Theme.bgPrimary)
        .ignoresSafeArea()
        .foregroundStyle(Theme.textPrimary)
    }

    @ViewBuilder
    private func page(_ name: AppModel.PageName) -> some View {
        switch name {
        case .server: ServerPage()
        case .models: ModelsPage()
        case .connect: ConnectPage()
        case .statistics: StatsPage()
        }
    }

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 0) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Syrinx").font(.system(size: 20, weight: .bold))
                Caption("Local models for any chat app")
            }
            .padding(EdgeInsets(top: 44, leading: 20, bottom: 18, trailing: 16))

            VStack(spacing: 2) {
                ForEach(AppModel.PageName.allCases) { name in
                    navButton(name)
                }
            }
            .padding(.horizontal, 10)

            Spacer()

            Divider().overlay(Theme.border)
            VStack(alignment: .leading, spacing: 3) {
                let (color, text) = Theme.describe(server.state)
                HStack(spacing: 8) {
                    Circle().fill(color).frame(width: 8, height: 8)
                    Text(text).font(.system(size: 12))
                        .foregroundStyle(server.state == .running ? Theme.textPrimary : Theme.textSecondary)
                }
                if let detail = stateDetail {
                    Text(detail).font(.system(size: 11)).foregroundStyle(Theme.textSecondary).lineLimit(1).truncationMode(.middle)
                }
            }
            .padding(EdgeInsets(top: 12, leading: 20, bottom: 14, trailing: 16))
        }
    }

    private var stateDetail: String? {
        let model = server.model.map { (($0 as NSString).lastPathComponent as NSString).deletingPathExtension }
        switch server.state {
        case .running: return server.address.map { "\(model ?? "") · port \($0.port ?? 0)" }
        case .loading: return model
        default: return nil
        }
    }

    private func navButton(_ name: AppModel.PageName) -> some View {
        let on = app.page == name
        return Button {
            app.page = name
        } label: {
            HStack(spacing: 12) {
                Image(systemName: name.symbol).font(.system(size: 13)).frame(width: 18)
                Text(name.rawValue).font(.system(size: 13.5))
                Spacer()
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .foregroundStyle(on ? Theme.textPrimary : Theme.textSecondary)
            .background(on ? Theme.accentSoft : Color.clear, in: RoundedRectangle(cornerRadius: 9))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}
