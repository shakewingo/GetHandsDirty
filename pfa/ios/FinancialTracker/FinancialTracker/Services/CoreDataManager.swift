import CoreData
import Foundation

class CoreDataManager {
    static let shared = CoreDataManager()
    
    private init() {
        // Just print the database location for debugging, no manipulation here
        if let storeURL = persistentContainer.persistentStoreDescriptions.first?.url {
            print("CoreData database location: \(storeURL.path)")
        }
    }
    
    lazy var persistentContainer: NSPersistentContainer = {
        let container = NSPersistentContainer(name: "FinancialTracker")
        
        // Configure the persistent store description for automatic migrations
        if let description = container.persistentStoreDescriptions.first {
            description.shouldMigrateStoreAutomatically = true
            description.shouldInferMappingModelAutomatically = true
            
            // Add options using the proper setOption method
            description.setOption(FileProtectionType.complete as NSObject, forKey: NSPersistentStoreFileProtectionKey)
            description.setOption(true as NSNumber, forKey: NSPersistentStoreRemoteChangeNotificationPostOptionKey)
        }
        
        container.loadPersistentStores { description, error in
            if let error = error {
                print("Error loading persistent stores: \(error)")
                
                // Instead of immediately deleting, try recovery options
                if let storeURL = description.url, FileManager.default.fileExists(atPath: storeURL.path) {
                    // Create a backup of the problematic store
                    let backupURL = storeURL.deletingLastPathComponent().appendingPathComponent("FinancialTracker-backup.sqlite")
                    do {
                        if FileManager.default.fileExists(atPath: backupURL.path) {
                            try FileManager.default.removeItem(at: backupURL)
                        }
                        try FileManager.default.copyItem(at: storeURL, to: backupURL)
                        print("Created backup of store at \(backupURL.path)")
                        
                        // Now attempt to remove and recreate the store
                        try container.persistentStoreCoordinator.destroyPersistentStore(at: storeURL, ofType: description.type, options: description.options)
                        print("Successfully destroyed corrupted store")
                        
                        // Try loading again with a fresh store
                        try container.persistentStoreCoordinator.addPersistentStore(ofType: description.type, configurationName: description.configuration, at: description.url, options: description.options)
                        print("Successfully recreated store")
                    } catch {
                        // If all else fails, report the error but don't crash
                        print("Recovery failed: \(error)")
                    }
                }
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
                entity.id = Int32(asset.id)
                entity.assetType = asset.assetType
                entity.marketValueDouble = asset.marketValue
                entity.marketShareDouble = asset.marketShare
                entity.currency = asset.currency
                entity.createdAt = asset.createdAt
            }
            
            saveContext()
        } catch {
            print("Error saving assets: \(error)")
        }
    }
    
    func fetchAssets() -> [Asset] {
        let request = NSFetchRequest<AssetEntity>(entityName: "AssetEntity")
        
        do {
            let result = try context.fetch(request)
            return result.map { entity in
                Asset(id: Int(entity.id),
                     assetType: entity.assetType ?? "",
                     marketValue: entity.marketValueDouble,
                     marketShare: entity.marketShareDouble,
                     currency: entity.currency ?? "USD",
                     createdAt: entity.createdAt ?? "")
            }
        } catch {
            print("Error fetching assets: \(error)")
            return []
        }
    }
    
    func updateAsset(_ asset: Asset) {
        let fetchRequest: NSFetchRequest<AssetEntity> = AssetEntity.fetchRequest()
        fetchRequest.predicate = NSPredicate(format: "id == %d", asset.id)
        
        do {
            let results = try context.fetch(fetchRequest)
            if let entity = results.first {
                entity.assetType = asset.assetType
                entity.marketValueDouble = asset.marketValue
                entity.marketShareDouble = asset.marketShare
                entity.currency = asset.currency
                entity.createdAt = asset.createdAt
                saveContext()
            }
        } catch {
            print("Error updating asset: \(error)")
        }
    }
    
    func deleteAsset(id: Int) {
        let request = NSFetchRequest<AssetEntity>(entityName: "AssetEntity")
        request.predicate = NSPredicate(format: "id == %d", id)
        
        do {
            let result = try context.fetch(request)
            if let entity = result.first {
                context.delete(entity)
                saveContext()
            }
        } catch {
            print("Error deleting asset: \(error)")
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
                entity.id = Int32(credit.id)
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
        let request = NSFetchRequest<CreditEntity>(entityName: "CreditEntity")
        
        do {
            let result = try context.fetch(request)
            return result.map { entity in
                Credit(id: Int(entity.id),
                      creditType: entity.creditType ?? "",
                      marketValue: entity.marketValue,
                      currency: entity.currency ?? "USD",
                      createdAt: entity.createdAt ?? "")
            }
        } catch {
            print("Error fetching credits: \(error)")
            return []
        }
    }
    
    func updateCredit(_ credit: Credit) {
        let fetchRequest: NSFetchRequest<CreditEntity> = CreditEntity.fetchRequest()
        fetchRequest.predicate = NSPredicate(format: "id == %d", credit.id)
        
        do {
            let results = try context.fetch(fetchRequest)
            if let entity = results.first {
                entity.creditType = credit.creditType
                entity.marketValue = credit.marketValue
                entity.currency = credit.currency
                entity.createdAt = credit.createdAt
                saveContext()
            }
        } catch {
            print("Error updating credit: \(error)")
        }
    }
    
    func deleteCredit(id: Int) {
        let request = NSFetchRequest<CreditEntity>(entityName: "CreditEntity")
        request.predicate = NSPredicate(format: "id == %d", id)
        
        do {
            let result = try context.fetch(request)
            if let entity = result.first {
                context.delete(entity)
                saveContext()
            }
        } catch {
            print("Error deleting credit: \(error)")
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
                entity.id = Int32(0) // You might want to generate a proper ID here
                entity.date = transaction.date
                entity.desc = transaction.description
                entity.amount = transaction.amount
                entity.category = transaction.category
                entity.type = transaction.type
                entity.source = transaction.source
                entity.createdAt = Date()
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
