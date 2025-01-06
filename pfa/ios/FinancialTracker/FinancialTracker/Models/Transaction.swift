//
//  Transaction.swift
//  FinancialTracker
//
//  Created by Ying Yao on 2024/12/28.
//

import Foundation

struct Transaction: Codable, Identifiable {
    let date: String
    let description: String
    let amount: Double
    let category: String
    let type: String
    let source: String
    
    var id: String {
        // Create a unique identifier from the transaction properties
        "\(date)_\(description)_\(amount)"
    }
    
    enum CodingKeys: String, CodingKey {
        case date
        case description
        case amount
        case category
        case type
        case source
    }
}
