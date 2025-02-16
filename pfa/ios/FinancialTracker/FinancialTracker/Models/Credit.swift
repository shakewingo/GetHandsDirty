import Foundation

struct Credit: Identifiable, Codable {
    var id: Int = 0
    let creditType: String
    let marketValue: Double
    let currency: String
    let createdAt: String
    
    private enum CodingKeys: String, CodingKey {
        case id, creditType, marketValue, currency, createdAt
    }
}
