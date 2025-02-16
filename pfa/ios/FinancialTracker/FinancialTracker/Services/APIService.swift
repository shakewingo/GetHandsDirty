import Foundation

enum APIError: Error {
    case invalidURL
    case networkError(Error)
    case decodingError(Error)
    case serverError(Int)
    case noData
    
    var localizedDescription: String {
        switch self {
        case .invalidURL:
            return "Invalid URL"
        case .networkError(let error):
            return "Network error: \(error.localizedDescription)"
        case .decodingError(let error):
            return "Failed to decode response: \(error.localizedDescription)"
        case .serverError(let code):
            return "Server error with code: \(code)"
        case .noData:
            return "No data received from server"
        }
    }
}

class APIService {
    static let shared = APIService()
    
    #if targetEnvironment(simulator)
    private let baseURL = "http://localhost:8000/api"
    #else
    private let baseURL = "http://127.0.0.1:8000/api"
    #endif
    
    private init() {}
    
    private func handleResponse(_ response: URLResponse) throws -> HTTPURLResponse {
        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.networkError(NSError(domain: "", code: -1, userInfo: [NSLocalizedDescriptionKey: "Invalid response"]))
        }
        
        guard (200...299).contains(httpResponse.statusCode) else {
            throw APIError.serverError(httpResponse.statusCode)
        }
        
        return httpResponse
    }
    
    func fetchSummary() async throws -> AccountSummary {
        guard let url = URL(string: "\(baseURL)/summary") else {
            throw APIError.invalidURL
        }
        
        let (data, response) = try await URLSession.shared.data(from: url)
        _ = try handleResponse(response)
        
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(AccountSummary.self, from: data)
    }
    
    func fetchGroupedAssets() async throws -> [Asset] {
        guard let url = URL(string: "\(baseURL)/grouped_assets") else {
            throw APIError.invalidURL
        }
        
        let (data, response) = try await URLSession.shared.data(from: url)
        _ = try handleResponse(response)
        
        print("Raw API response data: \(String(data: data, encoding: .utf8) ?? "none")")
        
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let assets = try decoder.decode([Asset].self, from: data)
        print("Decoded assets: \(assets)")
        return assets
    }
    
    func fetchGroupedCredits() async throws -> [Credit] {
        guard let url = URL(string: "\(baseURL)/grouped_credits") else {
            throw APIError.invalidURL
        }
        
        let (data, response) = try await URLSession.shared.data(from: url)
        _ = try handleResponse(response)
        
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode([Credit].self, from: data)
    }
    
    func addAsset(assetType: String, marketValue: Double, currency: String) async throws -> Asset {
        guard let url = URL(string: "\(baseURL)/assets") else {
            throw APIError.invalidURL
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let asset = Asset(
            assetType: assetType,
            marketValue: marketValue,
            currency: currency,
            createdAt: ""
        )
        
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        request.httpBody = try encoder.encode(asset)
        
        let (data, response) = try await URLSession.shared.data(for: request)
        _ = try handleResponse(response)
        
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(Asset.self, from: data)
    }
    
    func addCredit(creditType: String, marketValue: Double, currency: String) async throws -> Credit {
        guard let url = URL(string: "\(baseURL)/credits") else {
            throw APIError.invalidURL
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let credit = Credit(
            creditType: creditType,
            marketValue: marketValue,
            currency: currency,
            createdAt: ""
        )
        
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        request.httpBody = try encoder.encode(credit)
        
        let (data, response) = try await URLSession.shared.data(for: request)
        _ = try handleResponse(response)
        
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(Credit.self, from: data)
    }
    
    func uploadStatements(files: [(URL, String)]) async throws -> [Transaction] {
        guard let url = URL(string: "\(baseURL)/upload_statements") else {
            throw APIError.invalidURL
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        
        let boundary = UUID().uuidString
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        
        var data = Data()
        
        for (fileURL, sourceType) in files {
            data.append("--\(boundary)\r\n".data(using: .utf8)!)
            data.append("Content-Disposition: form-data; name=\"files\"; filename=\"\(fileURL.lastPathComponent)\"\r\n".data(using: .utf8)!)
            data.append("Content-Type: application/octet-stream\r\n\r\n".data(using: .utf8)!)
            data.append(try Data(contentsOf: fileURL))
            data.append("\r\n".data(using: .utf8)!)
        }
        
        data.append("--\(boundary)\r\n".data(using: .utf8)!)
        data.append("Content-Disposition: form-data; name=\"source_types\"\r\n\r\n".data(using: .utf8)!)
        data.append(files.map { $0.1 }.joined(separator: ",").data(using: .utf8)!)
        data.append("\r\n".data(using: .utf8)!)
        data.append("--\(boundary)--\r\n".data(using: .utf8)!)
        
        request.httpBody = data
        
        let (responseData, response) = try await URLSession.shared.data(for: request)
        _ = try handleResponse(response)
        
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let uploadResponse = try decoder.decode(UploadResponse.self, from: responseData)
        return uploadResponse.transactions
    }
    
    func fetchAssetDetails(assetType: String, currency: String) async throws -> [Asset] {
        guard let encodedType = assetType.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed),
              let encodedCurrency = currency.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed),
              let url = URL(string: "\(baseURL)/asset_details?asset_type=\(encodedType)&currency=\(encodedCurrency)") else {
            throw APIError.invalidURL
        }
        
        let (data, response) = try await URLSession.shared.data(from: url)
        _ = try handleResponse(response)
        
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode([Asset].self, from: data)
    }
    
    func fetchCreditDetails(creditType: String, currency: String) async throws -> [Credit] {
        guard let encodedType = creditType.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed),
              let encodedCurrency = currency.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed),
              let url = URL(string: "\(baseURL)/credit_details?credit_type=\(encodedType)&currency=\(encodedCurrency)") else {
            throw APIError.invalidURL
        }
        
        let (data, response) = try await URLSession.shared.data(from: url)
        _ = try handleResponse(response)
        
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode([Credit].self, from: data)
    }
    
    func updateAsset(_ asset: Asset) async throws -> Asset {
        guard let url = URL(string: "\(baseURL)/assets/\(asset.id)") else {
            throw APIError.invalidURL
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "PUT"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        request.httpBody = try encoder.encode(asset)
        
        let (data, response) = try await URLSession.shared.data(for: request)
        _ = try handleResponse(response)
        
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(Asset.self, from: data)
    }
    
    func deleteAsset(id: String) async throws {
        guard let url = URL(string: "\(baseURL)/assets/\(id)") else {
            throw APIError.invalidURL
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "DELETE"
        
        let (_, response) = try await URLSession.shared.data(for: request)
        _ = try handleResponse(response)
    }
    
    func updateCredit(_ credit: Credit) async throws -> Credit {
        guard let url = URL(string: "\(baseURL)/credits/\(credit.id)") else {
            throw APIError.invalidURL
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "PUT"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        request.httpBody = try encoder.encode(credit)
        
        let (data, response) = try await URLSession.shared.data(for: request)
        _ = try handleResponse(response)
        
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(Credit.self, from: data)
    }
    
    func deleteCredit(id: String) async throws {
        guard let url = URL(string: "\(baseURL)/credits/\(id)") else {
            throw APIError.invalidURL
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "DELETE"
        
        let (_, response) = try await URLSession.shared.data(for: request)
        _ = try handleResponse(response)
    }
}

struct UploadResponse: Codable {
    let transactions: [Transaction]
}
