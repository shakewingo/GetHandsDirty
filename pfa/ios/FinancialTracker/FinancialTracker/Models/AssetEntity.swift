import Foundation
import CoreData

@objc(AssetEntity)
public class AssetEntity: NSManagedObject, Identifiable {

}

extension AssetEntity {
    @nonobjc public class func fetchRequest() -> NSFetchRequest<AssetEntity> {
        return NSFetchRequest<AssetEntity>(entityName: "AssetEntity")
    }
    
    @NSManaged public var id: Int32
    @NSManaged public var assetType: String?
    @NSManaged public var marketValue: Double
    @NSManaged public var currency: String?
    @NSManaged public var createdAt: String?
}
