    import XCTest
    @testable import SwiftLevenbergMarquardt

    final class SwiftLevenbergMarquardtTests: XCTestCase {
        func testMatrixOperations() {
            XCTAssert(testMatrixFunctions())
        }
        
        func testNewtonIteration() {
            let opt = NewtonIterationOptimizer()
            XCTAssert(solvesLinearEquation(optimizer: opt))
        }
        
        func testGradientDescent() {
            let opt = GradientDescentOptimizer()
            XCTAssert(solvesLinearEquation(optimizer: opt))
        }
        
        func testLevMarq() {
            let opt = LevenbergMarquardtOptimizer()
            XCTAssert(solvesLinearEquation(optimizer: opt))
        }
        
        func testLevMarqWithInputOutput() {
            let opt = LevenbergMarquardtOptimizer()
            XCTAssert(solvesLinearEquationWithIO(optimizer: opt))
        }
        
    }
