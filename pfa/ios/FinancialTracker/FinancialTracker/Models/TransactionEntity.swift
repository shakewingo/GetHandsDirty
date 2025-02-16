import Foundation
import CoreData

@objc(TransactionEntity)
public class TransactionEntity: NSManagedObject, Identifiable {

}

extension TransactionEntity {
    @nonobjc public class func fetchRequest() -> NSFetchRequest<TransactionEntity> {
        return NSFetchRequest<TransactionEntity>(entityName: "TransactionEntity")
    }
    
    @NSManaged public var id: Int32
    @NSManaged public var date: String?
    @NSManaged public var desc: String?
    @NSManaged public var amount: Double
    @NSManaged public var category: String?
    @NSManaged public var type: String?
    @NSManaged public var source: String?
    @NSManaged public var createdAt: Date?
} 