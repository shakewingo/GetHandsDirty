import Foundation

struct Credit: Identifiable, Codable {
    var id: Int = 0
    var creditType: String
    var marketValue: Double
    var currency: String
    var createdAt: String
    
    private enum CodingKeys: String, CodingKey {
        case id, creditType, marketValue, currency, createdAt
    }
}
