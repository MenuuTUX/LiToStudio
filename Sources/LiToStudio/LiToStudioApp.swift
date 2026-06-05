import SwiftUI

@main
struct LiToStudioApp: App {
    @State private var model = AppModel()

    var body: some Scene {
        WindowGroup("LiTo Studio") {
            ContentView()
                .environment(model)
                .frame(minWidth: 1060, minHeight: 700)
                .preferredColorScheme(.dark)
        }
        .windowResizability(.contentMinSize)
    }
}
