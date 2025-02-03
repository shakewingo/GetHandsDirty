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
    private let baseURL = "http://localhost:8000/api"  // Use this for simulator
    #else
    private let baseURL = "http://127.0.0.1:8000/api"   // Use this for physical device
    #endif
    
    private init() {
        print("APIService initialized with baseURL: \(baseURL)")
    }
    
    func fetchSummary() async throws -> AccountSummary {
        let urlString = "\(baseURL)/summary"
        print("Attempting to fetch summary from: \(urlString)")
        
        guard let url = URL(string: urlString) else {
            print("Invalid URL: \(urlString)")
            throw APIError.invalidURL
        }
        
        do {
            print("Starting network request...")
            let (data, response) = try await URLSession.shared.data(from: url)
            print("Received response: \(response)")
            
            guard let httpResponse = response as? HTTPURLResponse else {
                print("Invalid response type")
                throw APIError.networkError(NSError(domain: "", code: -1, userInfo: [NSLocalizedDescriptionKey: "Invalid response"]))
            }
            
            print("HTTP Status Code: \(httpResponse.statusCode)")
            guard (200...299).contains(httpResponse.statusCode) else {
                print("Error status code: \(httpResponse.statusCode)")
                throw APIError.serverError(httpResponse.statusCode)
            }
            
            // Debug: Print raw response
            if let jsonString = String(data: data, encoding: .utf8) {
                print("Raw API Response: \(jsonString)")
            }
            
            let decoder = JSONDecoder()
            decoder.keyDecodingStrategy = .convertFromSnakeCase
            
            let summary = try decoder.decode(AccountSummary.self, from: data)
            print("Successfully decoded summary: \(summary)")
            return summary
        } catch let error as DecodingError {
            print("Decoding Error: \(error)")
            throw APIError.decodingError(error)
        } catch {
            print("Network Error: \(error)")
            throw APIError.networkError(error)
        }
    }
    
    func uploadStatements(files: [(URL, String)]) async throws -> [Transaction] {
        let url = URL(string: "\(baseURL)/upload_statements")!
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        
        let boundary = UUID().uuidString
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        
        var data = Data()
        
        // Add files
        for (fileURL, sourceType) in files {
            data.append("--\(boundary)\r\n".data(using: .utf8)!)
            data.append("Content-Disposition: form-data; name=\"files\"; filename=\"\(fileURL.lastPathComponent)\"\r\n".data(using: .utf8)!)
            data.append("Content-Type: application/octet-stream\r\n\r\n".data(using: .utf8)!)
            data.append(try Data(contentsOf: fileURL))
            data.append("\r\n".data(using: .utf8)!)
        }
        
        // Add source types
        data.append("--\(boundary)\r\n".data(using: .utf8)!)
        data.append("Content-Disposition: form-data; name=\"source_types\"\r\n\r\n".data(using: .utf8)!)
        data.append(files.map { $0.1 }.joined(separator: ",").data(using: .utf8)!)
        data.append("\r\n".data(using: .utf8)!)
        
        data.append("--\(boundary)--\r\n".data(using: .utf8)!)
        
        request.httpBody = data
        
        let (responseData, _) = try await URLSession.shared.data(for: request)
        let response = try JSONDecoder().decode(UploadResponse.self, from: responseData)
        return response.transactions
    }
    
    func fetchAssets() async throws -> [Asset] {
        let url = URL(string: "\(baseURL)/assets")!
        let (data, _) = try await URLSession.shared.data(from: url)
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode([Asset].self, from: data)
    }
    
    func fetchGroupedAssets() async throws -> [Asset] {
        let url = URL(string: "\(baseURL)/grouped_assets")!
        let (data, _) = try await URLSession.shared.data(from: url)
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode([Asset].self, from: data)
    }
    
    func addAsset(assetType: String, marketValue: Double, currency: String) async throws -> Asset {
        let url = URL(string: "\(baseURL)/assets")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        // Create an Asset struct instead of dictionary
        let asset = Asset(
            assetType: assetType,
            marketValue: marketValue,
            currency: currency,
            createdAt: ""  // Let backend set this
        )
        
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        request.httpBody = try encoder.encode(asset)
        
        let (data, response) = try await URLSession.shared.data(for: request)
        
        // Debug logging
        if let httpResponse = response as? HTTPURLResponse {
            print("HTTP Status Code: \(httpResponse.statusCode)")
            if let responseString = String(data: data, encoding: .utf8) {
                print("Response data: \(responseString)")
            }
        }
        
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(Asset.self, from: data)
    }
    
    func fetchCredits() async throws -> [Credit] {
        let url = URL(string: "\(baseURL)/credits")!
        let (data, _) = try await URLSession.shared.data(from: url)
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode([Credit].self, from: data)
    }
    
    func fetchGroupedCredits() async throws -> [Credit] {
        let url = URL(string: "\(baseURL)/grouped_credits")!
        let (data, _) = try await URLSession.shared.data(from: url)
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode([Credit].self, from: data)
    }
    
    func addCredit(creditType: String, marketValue: Double, currency: String) async throws -> Credit {
        let url = URL(string: "\(baseURL)/credits")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let credit = Credit(
            creditType: creditType,
            marketValue: marketValue,
            currency: currency,
            createdAt: ""  // Let backend set this
        )
        
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        request.httpBody = try encoder.encode(credit)
        
        let (data, response) = try await URLSession.shared.data(for: request)
        
        // Debug logging
        if let httpResponse = response as? HTTPURLResponse {
            print("HTTP Status Code: \(httpResponse.statusCode)")
            if let responseString = String(data: data, encoding: .utf8) {
                print("Response data: \(responseString)")
            }
        }
        
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(Credit.self, from: data)
    }
    
//    func fetchAssetDetails(assetType: String, currency: String) async throws -> [Asset] {
//        var components = URLComponents(string: "\(baseURL)/assets_details")!
//        components.queryItems = [
//            URLQueryItem(name: "asset_type", value: assetType),
//            URLQueryItem(name: "currency", value: currency)
//        ]
//        
//        guard let url = components.url else {
//            throw APIError.invalidURL
//        }
//        
//        let (data, _) = try await URLSession.shared.data(from: url)
//        let decoder = JSONDecoder()
//        decoder.keyDecodingStrategy = .convertFromSnakeCase
//        return try decoder.decode([Asset].self, from: data)
//    }
//    
//    func fetchCreditDetails(creditType: String, currency: String) async throws -> [Credit] {
//        var components = URLComponents(string: "\(baseURL)/credit_details")!
//        components.queryItems = [
//            URLQueryItem(name: "credit_type", value: creditType),
//            URLQueryItem(name: "currency", value: currency)
//        ]
//        
//        guard let url = components.url else {
//            throw APIError.invalidURL
//        }
//        
//        let (data, _) = try await URLSession.shared.data(from: url)
//        let decoder = JSONDecoder()
//        decoder.keyDecodingStrategy = .convertFromSnakeCase
//        return try decoder.decode([Credit].self, from: data)
//    }
}

struct UploadResponse: Codable {
    let transactions: [Transaction]
}
