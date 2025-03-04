import Foundation
import CoreData

@objc(AssetEntity)
public class AssetEntity: NSManagedObject, Identifiable {
    @NSManaged public var id: Int32
    @NSManaged public var assetType: String?
    @NSManaged public var marketValue: NSNumber?
    @NSManaged public var marketShare: NSNumber?
    @NSManaged public var currency: String?
    @NSManaged public var createdAt: String?
}

extension AssetEntity {
    @nonobjc public class func fetchRequest() -> NSFetchRequest<AssetEntity> {
        return NSFetchRequest<AssetEntity>(entityName: "AssetEntity")
    }
    
    var marketValueDouble: Double? {
        get { return marketValue?.doubleValue }
        set { marketValue = newValue.map { NSNumber(value: $0) } }
    }
    
    var marketShareDouble: Double? {
        get { return marketShare?.doubleValue }
        set { marketShare = newValue.map { NSNumber(value: $0) } }
    }
}
