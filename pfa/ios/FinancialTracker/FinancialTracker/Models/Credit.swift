import Foundation

struct Credit: Identifiable, Codable {
    var id: UUID = UUID()
    let creditType: String
    let marketValue: Double
    let currency: String
    let createdAt: String
    
    private enum CodingKeys: String, CodingKey {
        case creditType, marketValue, currency, createdAt
    }
}
