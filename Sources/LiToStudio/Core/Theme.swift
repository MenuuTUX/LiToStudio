import SwiftUI

/// Central design system — dark, modern, creative-tool aesthetic.
enum Theme {
    static let bg      = Color(red: 0.06, green: 0.06, blue: 0.08)
    static let bg2     = Color(red: 0.09, green: 0.09, blue: 0.12)
    static let card    = Color(red: 0.12, green: 0.12, blue: 0.15)
    static let cardHi  = Color(red: 0.16, green: 0.16, blue: 0.20)
    static let stroke  = Color.white.opacity(0.08)
    static let accent  = Color(red: 0.58, green: 0.46, blue: 1.00)
    static let accent2 = Color(red: 0.32, green: 0.82, blue: 0.92)
    static let good    = Color(red: 0.35, green: 0.85, blue: 0.55)
    static let bad     = Color(red: 0.95, green: 0.42, blue: 0.45)
    static let dim     = Color.white.opacity(0.5)

    static var accentGradient: LinearGradient {
        LinearGradient(colors: [accent, accent2],
                       startPoint: .topLeading, endPoint: .bottomTrailing)
    }
    static var sceneGradient: LinearGradient {
        LinearGradient(colors: [bg2, bg], startPoint: .top, endPoint: .bottom)
    }
}

/// Rounded card container.
struct CardModifier: ViewModifier {
    var highlighted: Bool = false
    func body(content: Content) -> some View {
        content
            .background(highlighted ? Theme.cardHi : Theme.card,
                        in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .strokeBorder(Theme.stroke, lineWidth: 1)
            )
    }
}

extension View {
    func card(highlighted: Bool = false) -> some View { modifier(CardModifier(highlighted: highlighted)) }
}
