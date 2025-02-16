import Foundation

struct Asset: Identifiable, Codable {
    var id: Int = 0
    let assetType: String
    let marketValue: Double
    let currency: String
    let createdAt: String
    
    private enum CodingKeys: String, CodingKey {
        case id, assetType, marketValue, currency, createdAt
    }
} 
