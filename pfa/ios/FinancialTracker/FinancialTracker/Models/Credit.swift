import Foundation

struct Credit: Identifiable, Codable {
    var id: UUID = UUID()
    let creditType: String
    let marketValue: Double
    let currency: String
    let createdAt: String
    
    enum CodingKeys: String, CodingKey {
        case creditType = "credit_type"
        case marketValue = "market_value"
        case currency
        case createdAt = "created_at"
    }
}
