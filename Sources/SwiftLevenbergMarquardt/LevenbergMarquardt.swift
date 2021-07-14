import Foundation
import simd

let maxLMIterations = 10

/**
Does Levenberg-Marquardt optimizaiton on the provide dparameters
    
- Parameter f: function to minimize
- Parameter X: desired values of function
- Parameter P: parameters to optimize
 
 - Returns: optimized parameters
*/
public class LevenbergMarquardtOptimizer : Optimizer {
    public var tolerance: Double = 0.01
    public var max_iters: Int = 10_000
    public var max_retries: Int = 10
    public init() {
        
    }
    
    func calculateError(newX: [Double], gtX: [Double]) -> [Double] {
        let error = zip(newX, gtX).map{ $0 - $1 }
        return error
    }
    
    func calculateLambda(P: [Double], JtJ: [Double], l: Double) -> [Double] {
        var lambda: [Double] = [Double](repeating: 0.0, count: P.count * P.count)
        for j in 0..<P.count {
            lambda[j*P.count + j] = l * JtJ[j*P.count + j]
        }
        return lambda
    }
    
    func getNextP(f: OptFunc, X: [Double], P: [Double], error: [Double], lambdaMulti: Double) -> [Double] {
        // refine P
        let J = calculateJacobian(f: f, P: P)
        let Jt = transpose(A: J, M: X.count, N: P.count)
        let JtJ = inner(A: Jt, B: J, M: P.count, P: X.count, N: P.count)
        let minusJterror = inner(A: Jt, B: error, M: P.count, P: X.count, N: 1).map{$0 * -1.0}

        let lambda = calculateLambda(P: P, JtJ: JtJ, l: lambdaMulti)
        
        let sumelement = zip(JtJ, lambda).map{$0 + $1}

        let delta = solve(A: sumelement, b: minusJterror, AM: P.count, AN: P.count, bM: P.count, bN: 1)
        
        return zip(P, delta).map{$0 + $1}
    }
    
    func getNextP(f: OptFuncWithInputOutput, X: [Double], x: [Double], P: [Double], error: [Double], lambdaMulti: Double) -> [Double] {
        // refine P
        let J = calculateJacobian(f: f, P: P, x: x)
        let Jt = transpose(A: J, M: X.count, N: P.count)
        let JtJ = inner(A: Jt, B: J, M: P.count, P: X.count, N: P.count)
        let minusJterror = inner(A: Jt, B: error, M: P.count, P: X.count, N: 1).map{$0 * -1.0}

        let lambda = calculateLambda(P: P, JtJ: JtJ, l: lambdaMulti)
        
        let sumelement = zip(JtJ, lambda).map{$0 + $1}

        let delta = solve(A: sumelement, b: minusJterror, AM: P.count, AN: P.count, bM: P.count, bN: 1)
        
        return zip(P, delta).map{$0 + $1}
    }
    
    public func optimizeWithInputOutput(f: OptFuncWithInputOutput, X: [Double], P: [Double], x: [Double]) -> [Double] {
        var iter = 0
        var currP = P
        var Xp: [Double] = []
        var lambdaMulti = 1e-3
        var prevError = Double.greatestFiniteMagnitude
        var currError = 0.0
        var errorVec: [Double] = []
        
        while iter < max_iters {
            // check error
            Xp = f(currP, x)
            errorVec = calculateError(newX: Xp, gtX: X)
            prevError = sqrt(errorVec.reduce(0.0) { r, d in
                r + (d*d)
            })
            
            //print("\(iter)/\(max_iters) err: \(prevError)")
            if prevError < tolerance {
                // done
                //print("NI converged")
                return currP
            }
                        
            // thsi doesnt trigger on first iteration
            // if error didnt decrease then increase lambdaMulti adn try again
            currError = prevError + 1
            while currError >= prevError {
                let testP = getNextP(f: f, X: X, x: x, P: currP, error: errorVec, lambdaMulti: lambdaMulti)
                let testXp = f(testP, x)
                errorVec = calculateError(newX: testXp, gtX: X)
                currError = sqrt(errorVec.reduce(0.0) { r, d in
                    r + (d*d)
                })
                
                if currError < prevError {
                    currP = testP
                    lambdaMulti /= 10
                    break
                }
                lambdaMulti *= 10
                if lambdaMulti > 1e9 {
                    // failed
                    //print("LM early quit, lambda too large: \(lambdaMulti)")
                    return currP
                }
            }
            
            iter += 1
        }
        return currP
    }
    
    public func optimize(f: OptFunc, X: [Double], P: [Double]) -> [Double] {
        // calculate average error for each value
        //var error = [Double](repeating: 0.0, count: X.count)

        //for idx in 0..<error.count {
        //    error[idx] += Xp[idx] - X[idx]
        //}
            
        var iter = 0
        var currP = P
        var Xp: [Double] = []
        var lambdaMulti = 1e-3
        var prevError = Double.greatestFiniteMagnitude
        var currError = 0.0
        var errorVec: [Double] = []
        
        while iter < max_iters {
            // check error
            Xp = f(currP)
            errorVec = calculateError(newX: Xp, gtX: X)
            prevError = sqrt(errorVec.reduce(0.0) { r, d in
                r + (d*d)
            })
            
            //print("\(iter)/\(max_iters) err: \(prevError)")
            if prevError < tolerance {
                // done
                //print("NI converged")
                return currP
            }
                        
            // thsi doesnt trigger on first iteration
            // if error didnt decrease then increase lambdaMulti adn try again
            currError = prevError + 1
            while currError >= prevError {
                let testP = getNextP(f: f, X: X, P: currP, error: errorVec, lambdaMulti: lambdaMulti)
                let testXp = f(testP)
                errorVec = calculateError(newX: testXp, gtX: X)
                currError = sqrt(errorVec.reduce(0.0) { r, d in
                    r + (d*d)
                })
                
                if currError < prevError {
                    currP = testP
                    lambdaMulti /= 10
                    break
                }
                lambdaMulti *= 10
                if lambdaMulti > 1e9 {
                    // failed
                    //print("LM early quit, lambda too large: \(lambdaMulti)")
                    return currP
                }
            }
            
            iter += 1
        }
        return currP
    }
}
