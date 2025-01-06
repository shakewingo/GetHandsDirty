import SwiftUI

@MainActor
class FinancialViewModel: ObservableObject {
    @Published var transactions: [Transaction] = []
    @Published var selectedFiles: [(URL, String)] = []
    @Published var isLoading = false
    @Published var errorMessage: String?
    @Published var summary: AccountSummary?
    
    private let apiService = APIService.shared
    
    func uploadSelectedFiles() async {
        guard !selectedFiles.isEmpty else { return }
        
        isLoading = true
        errorMessage = nil
        
        do {
            let newTransactions = try await apiService.uploadStatements(files: selectedFiles)
            self.transactions = newTransactions
            self.selectedFiles = []  // Clear selected files after successful upload
            await fetchSummary()  // Refresh summary after upload
        } catch {
            errorMessage = error.localizedDescription
        }
        
        isLoading = false
    }
    
    func fetchSummary() async {
        isLoading = true
        errorMessage = nil
        
        do {
            summary = try await apiService.fetchSummary()
        } catch {
            errorMessage = error.localizedDescription
        }
        
        isLoading = false
    }
    
    func clearError() {
        errorMessage = nil
    }
    
    // Helper methods for formatting
    func formatCurrency(_ value: Double) -> String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .currency
        formatter.currencyCode = "CAD"
        formatter.maximumFractionDigits = 1
        formatter.minimumFractionDigits = 1
        return formatter.string(from: NSNumber(value: abs(value))) ?? "$0.0"
    }
} 
