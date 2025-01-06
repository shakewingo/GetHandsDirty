//
//  Item.swift
//  FinancialTracker
//
//  Created by Ying Yao on 2024/12/28.
//

struct Item: Codable, Identifiable {
    let id: Int
    let name: String
    let description: String
}
