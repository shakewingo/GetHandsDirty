import Foundation

struct AccountSummary: Codable {
    let totalAssets: Double
    let totalCredit: Double
    let netWorth: Double
    let monthlySummary: [String: [String: Double]]
}
