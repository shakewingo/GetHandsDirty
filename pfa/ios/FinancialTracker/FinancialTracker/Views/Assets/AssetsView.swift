import SwiftUI

struct AssetsView: View {
    @StateObject private var viewModel = FinancialViewModel()
    @State private var isShowingAddSheet = false
    @State private var selectedAddType: AddType? = nil
    @State private var showError = false
    @State private var canPresentSheet = true
    
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
                    HStack {
                        Button(action: {
                            Task {
                                await viewModel.fetchSummary()
                            }
                        }) {
                            Image(systemName: "arrow.clockwise")
                        }
                        
                        Menu {
                            Button(action: {
                                if canPresentSheet {
                                    selectedAddType = .asset
                                    isShowingAddSheet = true
                                }
                            }) {
                                Label("Add Asset", systemImage: "plus.circle")
                            }
                            
                            Button(action: {
                                if canPresentSheet {
                                    selectedAddType = .credit
                                    isShowingAddSheet = true
                                }
                            }) {
                                Label("Add Credit", systemImage: "plus.circle")
                            }
                        } label: {
                            Image(systemName: "plus")
                        }
                        .disabled(!canPresentSheet)
                    }
                }
            }
            .sheet(isPresented: $isShowingAddSheet, onDismiss: {
                canPresentSheet = false
                selectedAddType = nil
                Task {
                    await viewModel.fetchSummary()
                    try? await Task.sleep(nanoseconds: 300_000_000)
                    canPresentSheet = true
                }
            }) {
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
        .onChange(of: viewModel.errorMessage) { oldValue, newValue in
            if newValue != nil && !isShowingAddSheet {
                showError = true
            }
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

struct AssetGroup: Identifiable {
    let id: String
    let assetType: String
    let currency: String
    let totalValue: Double
    let assets: [Asset]
}

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
    @State private var selectedAsset: Asset?
    @State private var showEditAssetSheet = false
    @State private var showDeleteAssetAlert = false
    @State private var canPresentSheet = true
    
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
                                ForEach(viewModel.getAssetDetails(assetType: group.assetType, currency: group.currency)) { asset in
                                    VStack(alignment: .leading, spacing: 4) {
                                        HStack {
                                            VStack(alignment: .leading, spacing: 4) {
                                                Text(viewModel.formatCurrency(asset.marketValue, currency: asset.currency))
                                                    .font(.subheadline)
                                                    .foregroundColor(.green)
                                                Text(asset.createdAt)
                                                    .font(.caption)
                                                    .foregroundColor(.secondary)
                                            }
                                            Spacer()
                                            HStack(spacing: 16) {
                                                Button(action: {
                                                    if canPresentSheet {
                                                        showEditAssetSheet = true
                                                        selectedAsset = asset
                                                    }
                                                }) {
                                                    Image(systemName: "pencil")
                                                        .foregroundColor(.blue)
                                                }
                                                .disabled(!canPresentSheet)
                                                Button(action: {
                                                    showDeleteAssetAlert = true
                                                    selectedAsset = asset
                                                }) {
                                                    Image(systemName: "trash")
                                                        .foregroundColor(.red)
                                                }
                                            }
                                        }
                                    }
                                    .padding(.leading)
                                    if asset.id != viewModel.getAssetDetails(assetType: group.assetType, currency: group.currency).last?.id {
                                        Divider()
                                    }
                                }
                            }
                        }
                        .frame(maxHeight: 200)
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
        .sheet(isPresented: $showEditAssetSheet, onDismiss: {
            canPresentSheet = false
            selectedAsset = nil
            Task {
                try? await Task.sleep(nanoseconds: 300_000_000)
                canPresentSheet = true
            }
        }) {
            if let asset = selectedAsset {
                EditAssetView(viewModel: viewModel, asset: asset)
            }
        }
        .alert("Delete Asset", isPresented: $showDeleteAssetAlert) {
            Button("Cancel", role: .cancel) { }
            Button("Delete", role: .destructive) {
                if let asset = selectedAsset {
                    Task {
                        await viewModel.deleteAsset(id: asset.id)
                        await viewModel.fetchAssetDetails(assetType: group.assetType, currency: group.currency)
                    }
                }
            }
        } message: {
            Text("Are you sure you want to delete this asset?")
        }
    }
}

struct CreditGroupRow: View {
    let group: CreditGroup
    @ObservedObject var viewModel: FinancialViewModel
    @State private var isExpanded = false
    @State private var selectedCredit: Credit?
    @State private var showEditCreditSheet = false
    @State private var showDeleteCreditAlert = false
    
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
                                ForEach(viewModel.getCreditDetails(creditType: group.creditType, currency: group.currency)) { credit in
                                    VStack(alignment: .leading, spacing: 4) {
                                        HStack {
                                            VStack(alignment: .leading, spacing: 4) {
                                                Text(viewModel.formatCurrency(credit.marketValue, currency: credit.currency))
                                                    .font(.subheadline)
                                                    .foregroundColor(.red)
                                                Text(credit.createdAt)
                                                    .font(.caption)
                                                    .foregroundColor(.secondary)
                                            }
                                            Spacer()
                                            HStack(spacing: 16) {
                                                Button(action: {
                                                    showEditCreditSheet = true
                                                    selectedCredit = credit
                                                }) {
                                                    Image(systemName: "pencil")
                                                        .foregroundColor(.blue)
                                                }
                                                Button(action: {
                                                    showDeleteCreditAlert = true
                                                    selectedCredit = credit
                                                }) {
                                                    Image(systemName: "trash")
                                                        .foregroundColor(.red)
                                                }
                                            }
                                        }
                                    }
                                    .padding(.leading)
                                    if credit.id != viewModel.getCreditDetails(creditType: group.creditType, currency: group.currency).last?.id {
                                        Divider()
                                    }
                                }
                            }
                        }
                        .frame(maxHeight: 200)
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
        .fullScreenCover(item: $selectedCredit, onDismiss: {
            selectedCredit = nil
            Task {
                await viewModel.fetchCreditDetails(creditType: group.creditType, currency: group.currency)
            }
        }) { credit in
            EditCreditView(viewModel: viewModel, credit: credit)
        }
        .alert("Delete Credit", isPresented: $showDeleteCreditAlert) {
            Button("Cancel", role: .cancel) { }
            Button("Delete", role: .destructive) {
                if let credit = selectedCredit {
                    Task {
                        await viewModel.deleteCredit(id: credit.id)
                        await viewModel.fetchCreditDetails(creditType: group.creditType, currency: group.currency)
                    }
                }
            }
        } message: {
            Text("Are you sure you want to delete this credit?")
        }
    }
}

