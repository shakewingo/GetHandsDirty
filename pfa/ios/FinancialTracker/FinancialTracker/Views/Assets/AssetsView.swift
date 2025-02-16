import SwiftUI

struct AssetsView: View {
    @StateObject private var viewModel = FinancialViewModel()
    @State private var isShowingAddSheet = false
    @State private var selectedAddType: AddType? = nil
    @State private var showError = false
    
    enum AddType {
        case asset
        case credit
    }
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Summary Cards
                    if let summary = viewModel.summary {
                        HStack(spacing: 16) {
                            ValueItem(
                                title: "Total Assets",
                                value: viewModel.formatCurrency(summary.totalAssets),
                                color: .green
                            )
                            
                            ValueItem(
                                title: "Total Credit",
                                value: viewModel.formatCurrency(summary.totalCredit),
                                color: .red
                            )
                        }
                        
                        ValueItem(
                            title: "Net Assets",
                            value: viewModel.formatCurrency(summary.netWorth),
                            color: summary.netWorth >= 0 ? .green : .red
                        )
                    }
                    
                    // Assets List
                    if !viewModel.assets.isEmpty {
                        VStack(alignment: .leading, spacing: 12) {
                            Text("Your Assets")
                                .font(.headline)
                                .padding(.horizontal)
                            
                            LazyVStack(spacing: 12) {
                                ForEach(groupAssets(viewModel.assets)) { group in
                                    AssetGroupRow(group: group, viewModel: viewModel)
                                }
                            }
                            .padding(.horizontal)
                        }
                    }
                    
                    // Credits List
                    if !viewModel.credits.isEmpty {
                        VStack(alignment: .leading, spacing: 12) {
                            Text("Your Credits")
                                .font(.headline)
                                .padding(.horizontal)
                            
                            LazyVStack(spacing: 12) {
                                ForEach(groupCredits(viewModel.credits)) { group in
                                    CreditGroupRow(group: group, viewModel: viewModel)
                                }
                            }
                            .padding(.horizontal)
                        }
                    }
                }
                .padding()
            }
            .navigationTitle("Assets")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Menu {
                        Button(action: {
                            selectedAddType = .asset
                            isShowingAddSheet = true
                        }) {
                            Label("Add Asset", systemImage: "plus.circle")
                        }
                        
                        Button(action: {
                            selectedAddType = .credit
                            isShowingAddSheet = true
                        }) {
                            Label("Add Credit", systemImage: "plus.circle")
                        }
                    } label: {
                        Image(systemName: "plus")
                    }
                }
            }
            .sheet(isPresented: $isShowingAddSheet) {
                if selectedAddType == .asset {
                    AddAssetView(viewModel: viewModel)
                } else if selectedAddType == .credit {
                    AddCreditView(viewModel: viewModel)
                }
            }
            .refreshable {
                await viewModel.fetchSummary()
            }
        }
        .onAppear {
            Task {
                await viewModel.fetchSummary()
            }
        }
        .onChange(of: viewModel.errorMessage) { newValue in
            showError = newValue != nil
        }
        .alert("Error", isPresented: $showError) {
            Button("OK") {
                viewModel.clearError()
            }
        } message: {
            if let errorMessage = viewModel.errorMessage {
                Text(errorMessage)
            }
        }
    }
    
    // Helper function to group assets by type and currency
    private func groupAssets(_ assets: [Asset]) -> [AssetGroup] {
        Dictionary(grouping: assets) { asset in
            "\(asset.assetType)|\(asset.currency)"
    }
        .map { key, assets in
            let components = key.split(separator: "|")
            let totalValue = assets.reduce(0) { $0 + $1.marketValue }
            return AssetGroup(
                id: key,
                assetType: String(components[0]),
                currency: String(components[1]),
                totalValue: totalValue,
                assets: assets
            )
        }
        .sorted { $0.totalValue > $1.totalValue }
    }
    
    // Helper function to group credits by type and currency
    private func groupCredits(_ credits: [Credit]) -> [CreditGroup] {
        Dictionary(grouping: credits) { credit in
            "\(credit.creditType)|\(credit.currency)"
        }
        .map { key, credits in
            let components = key.split(separator: "|")
            let totalValue = credits.reduce(0) { $0 + $1.marketValue }
            return CreditGroup(
                id: key,
                creditType: String(components[0]),
                currency: String(components[1]),
                totalValue: totalValue,
                credits: credits
            )
        }
        .sorted { $0.totalValue > $1.totalValue }
    }
}

// Model for grouped assets
struct AssetGroup: Identifiable {
    let id: String
    let assetType: String
    let currency: String
    let totalValue: Double
    let assets: [Asset]
}

// Model for grouped credits
struct CreditGroup: Identifiable {
    let id: String
    let creditType: String
    let currency: String
    let totalValue: Double
    let credits: [Credit]
}

struct AssetGroupRow: View {
    let group: AssetGroup
    @ObservedObject var viewModel: FinancialViewModel
    @State private var isExpanded = false
    
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Button(action: {
                isExpanded.toggle()
                if isExpanded {
                    Task {
                        await viewModel.fetchAssetDetails(assetType: group.assetType, currency: group.currency)
                    }
                } else {
                    viewModel.clearSelectedDetails()
                }
            }) {
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text(group.assetType)
                            .font(.headline)
                        Spacer()
                        Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                            .foregroundColor(.secondary)
                    }
                    HStack {
                        Text(viewModel.formatCurrency(group.totalValue, currency: group.currency))
                            .font(.subheadline)
                            .foregroundColor(.green)
                        Spacer()
                        Text(group.currency)
                            .font(.caption)
                            .padding(4)
                            .background(Color.secondary.opacity(0.2))
                            .cornerRadius(4)
                    }
                }
            }
            .buttonStyle(PlainButtonStyle())
            
            if isExpanded {
                VStack(alignment: .leading, spacing: 8) {
                    Divider()
                    if viewModel.isLoading {
                        ProgressView()
                            .padding()
                    } else {
                        ScrollView {
                            LazyVStack(alignment: .leading, spacing: 8) {
                                ForEach(viewModel.selectedAssetDetails) { asset in
                                    VStack(alignment: .leading, spacing: 4) {
                                        Text(viewModel.formatCurrency(asset.marketValue, currency: asset.currency))
                                            .font(.subheadline)
                                            .foregroundColor(.green)
                                        Text(asset.createdAt)
                                            .font(.caption)
                                            .foregroundColor(.secondary)
                                    }
                                    .padding(.leading)
                                    if asset.id != viewModel.selectedAssetDetails.last?.id {
                                        Divider()
                                    }
                                }
                            }
                        }
                        .frame(maxHeight: 200) // Limit the height of the details scroll view
                    }
                }
                .transition(.opacity)
                .animation(.easeInOut, value: isExpanded)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(10)
        .shadow(radius: 2)
    }
}

struct CreditGroupRow: View {
    let group: CreditGroup
    @ObservedObject var viewModel: FinancialViewModel
    @State private var isExpanded = false
    
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Button(action: {
                isExpanded.toggle()
                if isExpanded {
                    Task {
                        await viewModel.fetchCreditDetails(creditType: group.creditType, currency: group.currency)
                    }
                } else {
                    viewModel.clearSelectedDetails()
                }
            }) {
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text(group.creditType)
                            .font(.headline)
                        Spacer()
                        Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                            .foregroundColor(.secondary)
                    }
                    HStack {
                        Text(viewModel.formatCurrency(group.totalValue, currency: group.currency))
                            .font(.subheadline)
                            .foregroundColor(.red)
                        Spacer()
                        Text(group.currency)
                            .font(.caption)
                            .padding(4)
                            .background(Color.secondary.opacity(0.2))
                            .cornerRadius(4)
                    }
                }
            }
            .buttonStyle(PlainButtonStyle())
            
            if isExpanded {
                VStack(alignment: .leading, spacing: 8) {
                    Divider()
                    if viewModel.isLoading {
                        ProgressView()
                            .padding()
                    } else {
                        ScrollView {
                            LazyVStack(alignment: .leading, spacing: 8) {
                                ForEach(viewModel.selectedCreditDetails) { credit in
                                    VStack(alignment: .leading, spacing: 4) {
                                        Text(viewModel.formatCurrency(credit.marketValue, currency: credit.currency))
                                            .font(.subheadline)
                                            .foregroundColor(.red)
                                        Text(credit.createdAt)
                                            .font(.caption)
                                            .foregroundColor(.secondary)
                                    }
                                    .padding(.leading)
                                    if credit.id != viewModel.selectedCreditDetails.last?.id {
                                        Divider()
                                    }
                                }
                            }
                        }
                        .frame(maxHeight: 200) // Limit the height of the details scroll view
                    }
                }
                .transition(.opacity)
                .animation(.easeInOut, value: isExpanded)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(10)
        .shadow(radius: 2)
    }
}
