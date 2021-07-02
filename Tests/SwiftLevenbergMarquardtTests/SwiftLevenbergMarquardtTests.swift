    import XCTest
    @testable import SwiftLevenbergMarquardt

    final class SwiftLevenbergMarquardtTests: XCTestCase {
        func testLinear() {
            // This is an example of a functional test case.
            // Use XCTAssert and related functions to verify your tests produce the correct
            // results.
            //XCTAssertEqual(testestest().text, "Hello, World!")
            XCTAssert(lmSolvesLinearEquation())
        }
        
        func testNonlinear() {
            XCTAssert(lmSolvesNonlinearEquation())
        }
    }
