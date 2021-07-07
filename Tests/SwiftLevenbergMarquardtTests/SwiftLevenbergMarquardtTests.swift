    import XCTest
    @testable import SwiftLevenbergMarquardt

    final class SwiftLevenbergMarquardtTests: XCTestCase {
        func testMatrixOperations() {
            XCTAssert(testMatrixFunctions())
        }
        
        func testNewtonIteration() {
            XCTAssert(solvesLinearEquation(optimizer: newtonIteration))
        }
        
        func testGradientDescent() {
            XCTAssert(solvesLinearEquation(optimizer: gradientDescent))
        }
        
        func testLevMarq() {
            XCTAssert(solvesLinearEquation(optimizer: levenbergMarquardt))
        }
        
    }
