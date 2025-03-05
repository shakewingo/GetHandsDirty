//
//  Untitled.swift
//  FinancialTracker
//
//  Created by Ying Yao on 2024/12/28.
//

import SwiftUI

class ItemsViewModel: ObservableObject {
    @Published var items: [Item] = []
    @Published var isLoading = false
    @Published var errorMessage: String?
    
    // private let baseURL = "http://localhost:8000" // Change this to your server URL
    private let baseURL = "http://0.0.0.0:8000"
    
    func fetchItems() {
        isLoading = true
        
        guard let url = URL(string: "\(baseURL)/items") else {
            errorMessage = "Invalid URL"
            isLoading = false
            return
        }
        
        URLSession.shared.dataTask(with: url) { data, response, error in
            DispatchQueue.main.async {
                self.isLoading = false
                
                if let error = error {
                    self.errorMessage = error.localizedDescription
                    return
                }
                
                guard let data = data else {
                    self.errorMessage = "No data received"
                    return
                }
                
                do {
                    let items = try JSONDecoder().decode([Item].self, from: data)
                    self.items = items
                } catch {
                    self.errorMessage = "Failed to decode data"
                }
            }
        }.resume()
    }
}
