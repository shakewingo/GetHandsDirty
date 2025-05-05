import Foundation

struct Asset: Identifiable, Codable {
    var id: Int = 0
    var assetType: String
    var marketValue: Double?
    var marketShare: Double?
    var currency: String
    var createdAt: String
    
    private enum CodingKeys: String, CodingKey {
        case id, assetType, marketValue, marketShare, currency, createdAt
    }
} 
