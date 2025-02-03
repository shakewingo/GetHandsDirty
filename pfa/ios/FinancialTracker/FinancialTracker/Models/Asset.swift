import Foundation

struct Asset: Identifiable, Codable {
    var id: UUID = UUID()
    let assetType: String
    let marketValue: Double
    let currency: String
    let createdAt: String
    
    private enum CodingKeys: String, CodingKey {
        case assetType, marketValue, currency, createdAt
    }
} 