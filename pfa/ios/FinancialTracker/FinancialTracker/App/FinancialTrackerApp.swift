//
//  FinancialTrackerApp.swift
//  FinancialTracker
//
//  Created by Ying Yao on 2024/12/28.
//

import SwiftUI

@main
struct FinancialTrackerApp: App {
    let persistenceController = CoreDataManager.shared
    
    var body: some Scene {
        WindowGroup {
            TabView {
                BillsView()
                    .tabItem {
                        Label("Bills", systemImage: "dollarsign.circle")
                    }
                
                AssetsView()
                    .tabItem {
                        Label("Assets", systemImage: "chart.pie")
                    }
            }
            .environment(\.managedObjectContext, persistenceController.context)
        }
    }
}
