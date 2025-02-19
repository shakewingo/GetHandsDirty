import SwiftUI

@MainActor
class FinancialViewModel: ObservableObject {
    @Published var transactions: [Transaction] = []
    @Published var selectedFiles: [(URL, String)] = []
    @Published var isLoading = false
    @Published var errorMessage: String?
    @Published var summary: AccountSummary?
    @Published var assets: [Asset] = []
    @Published var credits: [Credit] = []
    @Published private var assetDetailsMap: [String: [Asset]] = [:]
    @Published private var creditDetailsMap: [String: [Credit]] = [:]
    
    private let apiService = APIService.shared
    private let coreDataManager = CoreDataManager.shared
    
    init() {
        // Load initial data from CoreData
        loadFromCoreData()
    }
    
    private func loadFromCoreData() {
        transactions = coreDataManager.fetchTransactions()
        assets = coreDataManager.fetchAssets()
        credits = coreDataManager.fetchCredits()
        print("Loaded from CoreData - Assets count: \(assets.count), Credits count: \(credits.count)")
        calculateLocalSummary()
    }
    
    func uploadSelectedFiles() async {
        guard !selectedFiles.isEmpty else { return }
        
        isLoading = true
        errorMessage = nil
        
        do {
            let newTransactions = try await apiService.uploadStatements(files: selectedFiles)
            self.transactions = newTransactions
            self.selectedFiles = []  // Clear selected files after successful upload
            
            // Save to CoreData
            coreDataManager.saveTransactions(newTransactions)
            
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
            // Clear any previous error since API call succeeded
            errorMessage = nil
            await fetchAssets()  // Fetch assets
            await fetchCredits() // Fetch credits
            print("After API fetch - Assets count: \(assets.count), Credits count: \(credits.count)")
        } catch let error as APIError {
            // Handle specific API errors
            switch error {
            case .networkError(_):
                // For network errors (like server not running), just load from CoreData silently
                loadFromCoreData()
            default:
                // For other API errors, show the error message
                errorMessage = error.localizedDescription
                loadFromCoreData()
            }
        } catch {
            // For unexpected errors, show the error message
            errorMessage = error.localizedDescription
            loadFromCoreData()
        }
        
        isLoading = false
    }
    
    func fetchAssets() async {
        do {
            let fetchedAssets = try await apiService.fetchGroupedAssets()
            print("fetchedAssets: \(fetchedAssets)")
            if !fetchedAssets.isEmpty {
                assets = fetchedAssets
                // Save to CoreData only if we got data from API
                coreDataManager.saveAssets(fetchedAssets)
            }
            // Clear any error since the call succeeded
            errorMessage = nil
        } catch {
            print("Error fetching assets: \(error)")
            // Don't show error message for asset fetching, just fall back to CoreData
            assets = coreDataManager.fetchAssets()
        }
    }
    
    func fetchCredits() async {
        do {
            let fetchedCredits = try await apiService.fetchGroupedCredits()
            if !fetchedCredits.isEmpty {
                credits = fetchedCredits
                // Save to CoreData only if we got data from API
                coreDataManager.saveCredits(fetchedCredits)
            }
            // Clear any error since the call succeeded
            errorMessage = nil
        } catch {
            // Don't show error message for credit fetching, just fall back to CoreData
            credits = coreDataManager.fetchCredits()
        }
    }
    
    func getAssetDetails(assetType: String, currency: String) -> [Asset] {
        return assetDetailsMap["\(assetType)|\(currency)"] ?? []
    }
    
    func getCreditDetails(creditType: String, currency: String) -> [Credit] {
        return creditDetailsMap["\(creditType)|\(currency)"] ?? []
    }
    
    func fetchAssetDetails(assetType: String, currency: String) async {
        isLoading = true
        errorMessage = nil
        
        do {
            let details = try await apiService.fetchAssetDetails(assetType: assetType, currency: currency)
            assetDetailsMap["\(assetType)|\(currency)"] = details
        } catch {
            errorMessage = error.localizedDescription
            assetDetailsMap["\(assetType)|\(currency)"] = []
        }
        
        isLoading = false
    }
    
    func fetchCreditDetails(creditType: String, currency: String) async {
        isLoading = true
        errorMessage = nil
        
        do {
            let details = try await apiService.fetchCreditDetails(creditType: creditType, currency: currency)
            creditDetailsMap["\(creditType)|\(currency)"] = details
        } catch {
            errorMessage = error.localizedDescription
            creditDetailsMap["\(creditType)|\(currency)"] = []
        }
        
        isLoading = false
    }

    private func calculateLocalSummary() {
        let totalAssets = assets.reduce(0) { $0 + $1.marketValue }
        let totalCredit = credits.reduce(0) { $0 + $1.marketValue }
        
        // Create monthly summary
        var monthlySummary: [String: [String: Double]] = [:]
        for transaction in transactions {
            let year = String(transaction.date.suffix(4))
            let month = String(transaction.date.prefix(2))
            let monthKey = "\(year)-\(month)"
            
            if monthlySummary[monthKey] == nil {
                monthlySummary[monthKey] = [:]
            }
            
            if monthlySummary[monthKey]?[transaction.category] == nil {
                monthlySummary[monthKey]?[transaction.category] = 0
            }
            
            monthlySummary[monthKey]?[transaction.category]? += transaction.amount
        }
        
        summary = AccountSummary(
            totalAssets: totalAssets,
            totalCredit: totalCredit,
            netWorth: totalAssets + totalCredit,
            monthlySummary: monthlySummary
        )
    }

    func clearSelectedDetails() {
        assetDetailsMap.removeAll()
        creditDetailsMap.removeAll()
    }

    func clearError() {
        errorMessage = nil
    }
    
    // Helper methods for formatting
    func formatCurrency(_ value: Double, currency: String = "RMB") -> String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .currency
        formatter.currencyCode = currency
        formatter.maximumFractionDigits = 0
        formatter.minimumFractionDigits = 0
        
        // Use appropriate currency symbols
        switch currency {
        case "RMB":
            formatter.currencySymbol = "¥"
        case "USD":
            formatter.currencySymbol = "$"
        case "EUR":
            formatter.currencySymbol = "€"
        case "CAD":
            formatter.currencySymbol = "C$"
        default:
            formatter.currencySymbol = currency
        }
        
        return formatter.string(from: NSNumber(value: abs(value))) ?? "0"
    }

    func addAsset(assetType: String, marketValue: Double, currency: String) async {
        isLoading = true
        errorMessage = nil
        
        do {
            let asset = try await apiService.addAsset(assetType: assetType, marketValue: marketValue, currency: currency)
            // Save to CoreData
            coreDataManager.saveAssets([asset])
            // Refresh assets list
            await fetchAssets()
        } catch {
            errorMessage = error.localizedDescription
        }
        
        isLoading = false
    }


    func addCredit(creditType: String, marketValue: Double, currency: String) async {
        isLoading = true
        errorMessage = nil
        
        do {
            let credit = try await apiService.addCredit(creditType: creditType, marketValue: marketValue, currency: currency)
            // Save to CoreData
            coreDataManager.saveCredits([credit])
            // Refresh credits list
            await fetchCredits()
        } catch {
            errorMessage = error.localizedDescription
        }
        
        isLoading = false
    }
    
    func updateAsset(_ asset: Asset) async {
        isLoading = true
        errorMessage = nil
        
        do {
            let updatedAsset = try await apiService.updateAsset(asset)
            if let index = assets.firstIndex(where: { $0.id == asset.id }) {
                assets[index] = updatedAsset
            }
            // Update CoreData
            coreDataManager.updateAsset(updatedAsset)
            
            // Update local summary without making network calls
            calculateLocalSummary()
        } catch {
            errorMessage = error.localizedDescription
        }
        
        isLoading = false
    }
    
    func deleteAsset(id: Int) async {
        do {
            try await apiService.deleteAsset(id: String(id))
            assets.removeAll { $0.id == id }
            coreDataManager.deleteAsset(id: id)
            
            // Update local summary without making network calls
            calculateLocalSummary()
        } catch {
            errorMessage = error.localizedDescription
        }
    }
    
    func updateCredit(_ credit: Credit) async {
        isLoading = true
        errorMessage = nil
        
        do {
            let updatedCredit = try await apiService.updateCredit(credit)
            if let index = credits.firstIndex(where: { $0.id == credit.id }) {
                credits[index] = updatedCredit
            }
            // Update CoreData
            coreDataManager.updateCredit(updatedCredit)
            
            // Update local summary without making network calls
            calculateLocalSummary()
        } catch {
            errorMessage = error.localizedDescription
        }
        
        isLoading = false
    }
    
    func deleteCredit(id: Int) async {
        do {
            try await apiService.deleteCredit(id: String(id))
            credits.removeAll { $0.id == id }
            coreDataManager.deleteCredit(id: id)
            
            // Update local summary without making network calls
            calculateLocalSummary()
        } catch {
            errorMessage = error.localizedDescription
        }
    }
} 

