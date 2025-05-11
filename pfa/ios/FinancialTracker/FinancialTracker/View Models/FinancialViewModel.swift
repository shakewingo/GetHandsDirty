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
    
    // Cache timestamps for tracking data freshness
    private var lastSummaryFetchTime: Date?
    private var lastAssetsFetchTime: Date?
    private var lastCreditsFetchTime: Date?
    private let cacheValidityInterval: TimeInterval = 5 // Cache valid for 60 seconds
    
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
        
        // Calculate summary locally if we have local data
        calculateLocalSummary()
        
        // Refresh from API if needed but don't block UI
        Task {
            if shouldRefreshFromAPI() {
                await fetchSummaryFromAPI()
            }
        }
    }
    
    // Calculate summary from local data instead of API call
    private func calculateLocalSummary() {
        let totalAssets = assets.reduce(0) { $0 + ($1.marketValue ?? 0) }
        let totalCredit = credits.reduce(0) { $0 + $1.marketValue }
        let netWorth = totalAssets - totalCredit
        
        // Create a basic summary with local data
        // Note: monthlySummary will be empty until API data arrives
        summary = AccountSummary(
            totalAssets: totalAssets,
            totalCredit: totalCredit,
            netWorth: netWorth,
            monthlySummary: [:]
        )
    }
    
    private func shouldRefreshFromAPI() -> Bool {
        guard let lastFetchTime = lastSummaryFetchTime else {
            return true // No previous fetch, should refresh
        }
        
        // Check if cache is still valid
        return Date().timeIntervalSince(lastFetchTime) > cacheValidityInterval
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
            
            // After uploading files, we need fresh data from API since the server state has changed
            await fetchSummaryFromAPI()
        } catch {
            errorMessage = error.localizedDescription
        }
        
        isLoading = false
    }
    
    // Main summary fetch function that provides cached data if available
    func fetchSummary() async {
        // If cache is still valid, just return without API call
        if !shouldRefreshFromAPI() {
            return
        }
        
        // Otherwise fetch fresh data from API
        await fetchSummaryFromAPI()
    }
    
    // Internal function to actually fetch from API
    private func fetchSummaryFromAPI() async {
        isLoading = true
        errorMessage = nil
        
        do {
            summary = try await apiService.fetchSummary()
            lastSummaryFetchTime = Date()
            
            // Clear any previous error since API call succeeded
            errorMessage = nil
            
            // Fetch updated assets and credits data
            await fetchAssetsFromAPI()
            await fetchCreditsFromAPI()
            
            print("After API fetch - Assets count: \(assets.count), Credits count: \(credits.count)")
        } catch let error as APIError {
            // Handle specific API errors
            switch error {
            case .networkError(let underlyingError):
                // For network errors, provide more context
                print("Network error: \(underlyingError)")
                
                // Check if it's a connection error (server not running or unreachable)
                if (underlyingError as NSError).code == NSURLErrorCannotConnectToHost ||
                   (underlyingError as NSError).code == NSURLErrorNotConnectedToInternet ||
                   (underlyingError as NSError).code == NSURLErrorTimedOut {
                    errorMessage = "Cannot connect to the server. Please check your network connection and make sure the server is running."
                    print("Server connection issue: \(underlyingError.localizedDescription)")
                } else {
                    // For other network errors, show a general message
                    errorMessage = "Network error: \(underlyingError.localizedDescription)"
                }
                
                // Still load from CoreData
                loadFromCoreData()
            case .invalidURL:
                errorMessage = "API configuration error: Invalid URL"
                loadFromCoreData()
            default:
                // For other API errors, show the error message
                errorMessage = "API Error: \(error.localizedDescription)"
                loadFromCoreData()
            }
        } catch {
            // For unexpected errors, show the error message
            errorMessage = "Unexpected error: \(error.localizedDescription)"
            print("Unexpected fetch error: \(error)")
            loadFromCoreData()
        }
        
        isLoading = false
    }
    
    func fetchAssets() async {
        if lastAssetsFetchTime == nil || Date().timeIntervalSince(lastAssetsFetchTime!) > cacheValidityInterval {
            await fetchAssetsFromAPI()
        }
    }
    
    private func fetchAssetsFromAPI() async {
        do {
            let fetchedAssets = try await apiService.fetchGroupedAssets()
            print("fetchedAssets: \(fetchedAssets)")
            if !fetchedAssets.isEmpty {
                assets = fetchedAssets
                lastAssetsFetchTime = Date()
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
        if lastCreditsFetchTime == nil || Date().timeIntervalSince(lastCreditsFetchTime!) > cacheValidityInterval {
            await fetchCreditsFromAPI()
        }
    }
    
    private func fetchCreditsFromAPI() async {
        do {
            let fetchedCredits = try await apiService.fetchGroupedCredits()
            if !fetchedCredits.isEmpty {
                credits = fetchedCredits
                lastCreditsFetchTime = Date()
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
        // Use cached details if we have them and they're less than 60 seconds old
        let detailsCacheKey = "\(assetType)|\(currency)"
        
        isLoading = true
        errorMessage = nil
        
        do {
            let details = try await apiService.fetchAssetDetails(assetType: assetType, currency: currency)
            assetDetailsMap[detailsCacheKey] = details
        } catch {
            errorMessage = error.localizedDescription
            assetDetailsMap[detailsCacheKey] = []
        }
        
        isLoading = false
    }
    
    func fetchCreditDetails(creditType: String, currency: String) async {
        // Use cached details if we have them and they're less than 60 seconds old
        let detailsCacheKey = "\(creditType)|\(currency)"
        
        isLoading = true
        errorMessage = nil
        
        do {
            let details = try await apiService.fetchCreditDetails(creditType: creditType, currency: currency)
            creditDetailsMap[detailsCacheKey] = details
        } catch {
            errorMessage = error.localizedDescription
            creditDetailsMap[detailsCacheKey] = []
        }
        
        isLoading = false
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

    func addAsset(assetType: String, marketValue: Double?, marketShare: Double?, currency: String) async {
        isLoading = true
        errorMessage = nil
        
        do {
            let asset = try await apiService.addAsset(
                assetType: assetType,
                marketValue: marketValue,
                marketShare: marketShare,
                currency: currency
            )
            // Save to CoreData
            coreDataManager.saveAssets([asset])
            
            // Update local assets list and recalculate summary
            var updatedAssets = assets
            updatedAssets.append(asset)
            assets = updatedAssets
            calculateLocalSummary()
            
            // Reset the last fetch time to force a refresh on next API call
            lastAssetsFetchTime = nil
            lastSummaryFetchTime = nil
            
            // Fetch fresh data in background
            Task {
                await fetchAssetsFromAPI()
            }
        } catch let error as APIError {
            errorMessage = "API Error: \(error.localizedDescription)"
            print("API Error: \(error)")
        } catch {
            errorMessage = "Unexpected error: \(error.localizedDescription)"
            print("Unexpected error: \(error)")
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
            
            // Update local credits list and recalculate summary
            var updatedCredits = credits
            updatedCredits.append(credit)
            credits = updatedCredits
            calculateLocalSummary()
            
            // Reset the last fetch time to force a refresh on next API call
            lastCreditsFetchTime = nil
            lastSummaryFetchTime = nil
            
            // Fetch fresh data in background
            Task {
                await fetchCreditsFromAPI()
            }
        } catch {
            errorMessage = error.localizedDescription
        }
        
        isLoading = false
    }
    
    func updateAsset(_ asset: Asset) async {
        isLoading = true
        errorMessage = nil
        
        // Update in CoreData first to make UI responsive
        if let index = assets.firstIndex(where: { $0.id == asset.id }) {
            // Use the asset directly now that its properties are mutable
            assets[index] = asset
            
            // Also update in any loaded details
            for (key, details) in assetDetailsMap {
                if let detailIndex = details.firstIndex(where: { $0.id == asset.id }) {
                    var updatedDetails = details
                    updatedDetails[detailIndex] = asset
                    assetDetailsMap[key] = updatedDetails
                }
            }
            
            // Update CoreData
            coreDataManager.updateAsset(asset)
            
            // Recalculate summary locally
            calculateLocalSummary()
            
            // Then update on the server in background
            Task {
                do {
                    let updatedAsset = try await apiService.updateAsset(asset)
                    
                    // Update local data with server response
                    if let index = assets.firstIndex(where: { $0.id == asset.id }) {
                        assets[index] = updatedAsset
                    }
                    
                    // Update CoreData with final server data
                    coreDataManager.updateAsset(updatedAsset)
                    
                    // Reset cache timestamps
                    lastAssetsFetchTime = nil
                    lastSummaryFetchTime = nil
                    
                    // Recalculate summary
                    calculateLocalSummary()
                } catch {
                    errorMessage = error.localizedDescription
                }
            }
        } else {
            errorMessage = "Asset not found in local data"
        }
        
        isLoading = false
    }
    
    func deleteAsset(id: Int) async {
        do {
            // Remove from local data first for immediate UI update
            assets.removeAll { $0.id == id }
            coreDataManager.deleteAsset(id: id)
            
            // Recalculate summary locally
            calculateLocalSummary()
            
            // Then delete from server
            try await apiService.deleteAsset(id: String(id))
            
            // Reset cache timestamps
            lastAssetsFetchTime = nil
            lastSummaryFetchTime = nil
            
            // Fetch fresh summary in background
            Task {
                await fetchSummaryFromAPI()
            }
        } catch {
            errorMessage = error.localizedDescription
        }
    }
    
    func updateCredit(_ credit: Credit) async {
        isLoading = true
        errorMessage = nil
        
        // Update in CoreData first to make UI responsive
        if let index = credits.firstIndex(where: { $0.id == credit.id }) {
            // Use the credit directly now that its properties are mutable
            credits[index] = credit
            
            // Also update in any loaded details
            for (key, details) in creditDetailsMap {
                if let detailIndex = details.firstIndex(where: { $0.id == credit.id }) {
                    var updatedDetails = details
                    updatedDetails[detailIndex] = credit
                    creditDetailsMap[key] = updatedDetails
                }
            }
            
            // Update CoreData
            coreDataManager.updateCredit(credit)
            
            // Recalculate summary locally
            calculateLocalSummary()
            
            // Then update on the server in background
            Task {
                do {
                    let updatedCredit = try await apiService.updateCredit(credit)
                    
                    // Update local data with server response
                    if let index = credits.firstIndex(where: { $0.id == credit.id }) {
                        credits[index] = updatedCredit
                    }
                    
                    // Update CoreData with final server data
                    coreDataManager.updateCredit(updatedCredit)
                    
                    // Reset cache timestamps
                    lastCreditsFetchTime = nil
                    lastSummaryFetchTime = nil
                    
                    // Recalculate summary
                    calculateLocalSummary()
                } catch {
                    errorMessage = error.localizedDescription
                }
            }
        } else {
            errorMessage = "Credit not found in local data"
        }
        
        isLoading = false
    }
    
    func deleteCredit(id: Int) async {
        do {
            // Remove from local data first for immediate UI update
            credits.removeAll { $0.id == id }
            coreDataManager.deleteCredit(id: id)
            
            // Recalculate summary locally
            calculateLocalSummary()
            
            // Then delete from server
            try await apiService.deleteCredit(id: String(id))
            
            // Reset cache timestamps
            lastCreditsFetchTime = nil
            lastSummaryFetchTime = nil
            
            // Fetch fresh summary in background
            Task {
                await fetchSummaryFromAPI()
            }
        } catch {
            errorMessage = error.localizedDescription
        }
    }
} 

