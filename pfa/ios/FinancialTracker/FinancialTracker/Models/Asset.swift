import Foundation

struct Asset: Identifiable, Codable {
    var id: UUID = UUID()
    let assetType: String
    let marketValue: Double
    let currency: String
    let createdAt: String
    
    enum CodingKeys: String, CodingKey {
        case assetType = "asset_type"
        case marketValue = "market_value"
        case currency
        case createdAt = "created_at"
    }
} 