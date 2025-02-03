import Foundation
import CoreData

@objc(CreditEntity)
public class CreditEntity: NSManagedObject, Identifiable {

}

extension CreditEntity {
    @nonobjc public class func fetchRequest() -> NSFetchRequest<CreditEntity> {
        return NSFetchRequest<CreditEntity>(entityName: "CreditEntity")
    }
    
    @NSManaged public var id: UUID?
    @NSManaged public var creditType: String?
    @NSManaged public var marketValue: Double
    @NSManaged public var currency: String?
    @NSManaged public var createdAt: String?
}