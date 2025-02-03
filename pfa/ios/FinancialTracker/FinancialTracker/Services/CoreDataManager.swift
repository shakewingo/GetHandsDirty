import CoreData
import Foundation

class CoreDataManager {
    static let shared = CoreDataManager()
    
    private init() {
        // Print the database location
        if let storeURL = persistentContainer.persistentStoreDescriptions.first?.url {
            print("CoreData database location: \(storeURL.path)")
        }
    }
    
    lazy var persistentContainer: NSPersistentContainer = {
        let container = NSPersistentContainer(name: "FinancialTracker")
        container.loadPersistentStores { description, error in
            if let error = error {
                fatalError("Unable to load persistent stores: \(error)")
            }
        }
        return container
    }()
    
    var context: NSManagedObjectContext {
        persistentContainer.viewContext
    }
    
    func saveContext() {
        if context.hasChanges {
            do {
                try context.save()
            } catch {
                print("Error saving context: \(error)")
            }
        }
    }
    
    // MARK: - Assets
    func saveAssets(_ assets: [Asset]) {
        // Clear existing assets
        let fetchRequest: NSFetchRequest<NSFetchRequestResult> = AssetEntity.fetchRequest()
        let deleteRequest = NSBatchDeleteRequest(fetchRequest: fetchRequest)
        
        do {
            try context.execute(deleteRequest)
            
            // Save new assets
            for asset in assets {
                let entity = AssetEntity(context: context)
                entity.id = asset.id
                entity.assetType = asset.assetType
                entity.marketValue = asset.marketValue
                entity.currency = asset.currency
                entity.createdAt = asset.createdAt
            }
            
            saveContext()
        } catch {
            print("Error saving assets: \(error)")
        }
    }
    
    func fetchAssets() -> [Asset] {
        let fetchRequest: NSFetchRequest<AssetEntity> = AssetEntity.fetchRequest()
        
        do {
            let entities = try context.fetch(fetchRequest)
            return entities.map { entity in
                Asset(id: entity.id ?? UUID(),
                     assetType: entity.assetType ?? "",
                     marketValue: entity.marketValue,
                     currency: entity.currency ?? "",
                     createdAt: entity.createdAt ?? "")
            }
        } catch {
            print("Error fetching assets: \(error)")
            return []
        }
    }
    
    // MARK: - Credits
    func saveCredits(_ credits: [Credit]) {
        // Clear existing credits
        let fetchRequest: NSFetchRequest<NSFetchRequestResult> = CreditEntity.fetchRequest()
        let deleteRequest = NSBatchDeleteRequest(fetchRequest: fetchRequest)
        
        do {
            try context.execute(deleteRequest)
            
            // Save new credits
            for credit in credits {
                let entity = CreditEntity(context: context)
                entity.id = credit.id
                entity.creditType = credit.creditType
                entity.marketValue = credit.marketValue
                entity.currency = credit.currency
                entity.createdAt = credit.createdAt
            }
            
            saveContext()
        } catch {
            print("Error saving credits: \(error)")
        }
    }
    
    func fetchCredits() -> [Credit] {
        let fetchRequest: NSFetchRequest<CreditEntity> = CreditEntity.fetchRequest()
        
        do {
            let entities = try context.fetch(fetchRequest)
            return entities.map { entity in
                Credit(id: entity.id ?? UUID(),
                      creditType: entity.creditType ?? "",
                      marketValue: entity.marketValue,
                      currency: entity.currency ?? "",
                      createdAt: entity.createdAt ?? "")
            }
        } catch {
            print("Error fetching credits: \(error)")
            return []
        }
    }
    
    // MARK: - Transactions
    func saveTransactions(_ transactions: [Transaction]) {
        // Clear existing transactions
        let fetchRequest: NSFetchRequest<NSFetchRequestResult> = TransactionEntity.fetchRequest()
        let deleteRequest = NSBatchDeleteRequest(fetchRequest: fetchRequest)
        
        do {
            try context.execute(deleteRequest)
            
            // Save new transactions
            for transaction in transactions {
                let entity = TransactionEntity(context: context)
                entity.id = UUID()
                entity.date = transaction.date
                entity.desc = transaction.description
                entity.amount = transaction.amount
                entity.category = transaction.category
                entity.type = transaction.type
                entity.source = transaction.source
            }
            
            saveContext()
        } catch {
            print("Error saving transactions: \(error)")
        }
    }
    
    func fetchTransactions() -> [Transaction] {
        let fetchRequest: NSFetchRequest<TransactionEntity> = TransactionEntity.fetchRequest()
        
        do {
            let entities = try context.fetch(fetchRequest)
            return entities.map { entity in
                Transaction(date: entity.date ?? "",
                          description: entity.desc ?? "",
                          amount: entity.amount,
                          category: entity.category ?? "",
                          type: entity.type ?? "",
                          source: entity.source ?? "")
            }
        } catch {
            print("Error fetching transactions: \(error)")
            return []
        }
    }
}
