//
//  File.swift
//  
//
//  Created by Tuukka on 2021/07/02.
//

import Foundation
import simd
@testable import SwiftLevenbergMarquardt

extension Array where Element == Double {
    func withNoise(variance: Double) -> [Double] {
        return map { el in
            el + Double.random(in: -1..<1) * variance
        }
    }
}

func normalize(arr: [Double]) -> [Double] {
    let mag = sqrt(arr.reduce(0.0, { res, val in
        res + val * val
    }))
    
    return arr.map{$0 / mag}
}

func solvesLinearEquation(optimizer: Optimizer) -> Bool {
    // prepare function to calculate
    // f(x) = p_0 * x_0 + p_1 * x_1
    func f(P: [Double]) -> [Double] {
        let a: Double = P[0]
        let b: Double = P[1]
        var ret: [Double] = []
        for idx in stride(from: 2, to: P.count, by: 2) {
            ret.append(a * P[idx])
            ret.append(b * P[idx+1])
        }
        //let a: Double = P[0] * P[1]
        //let b: Double = P[2] * P[3]
        return ret
    }

    // generate x, y pairs from f(x) = 123.45 * x + 987.65
    // with some noise
    var trueP: [Double] = [123.45, 987.65]
    
//    var Xs: [[Double]] = []
//    var Ys: [[Double]] = []
    
    for x in stride(from: 1.0, through: 100.0, by: 1.0) {
        trueP.append(x)
        trueP.append(1.0)
        //let inx = [x, 1.0]
        //let iny = f(P: trueP, X: inx).withNoise(variance: 100)
        //Xs.append(inx)
        //Ys.append(iny)
    }
    
    let outs = f(P: trueP)

    // parameters initial value
    //let params: [Double] = [0.0, 0.0]
    var params: [Double] = [100.0, 1000.0]
    
    for x in stride(from: 1.0, through: 100.0, by: 1.0) {
        params.append(x)// + Double.random(in: -0.1..<0.1))
        params.append(1.0)// + Double.random(in: -0.1..<0.1))
    }

    //let optP = optimizer(f: f, X: outs, P: params)
    let optP = optimizer.optimize(f: f, X: outs, P: params, mask: nil)

    print(optP)
    let ntp = normalize(arr: trueP)
    let nop = normalize(arr: optP)
    
    // calc dist to original params
    let dist: Double = sqrt(zip(ntp, nop).reduce(0.0) { res, val in
        res + (val.0 - val.1) * (val.0 - val.1)
    })
    
    print("dist: \(dist)")
    return dist < 0.1
    
}

func solvesLinearEquationWithIO(optimizer: Optimizer) -> Bool {
    // prepare function to calculate
    // f(x) = p_0 * x_0 + p_1 * x_1
    func f(P: [Double], x: [Double]) -> [Double] {
        let a: Double = P[0]
        let b: Double = P[1]
        var ret: [Double] = []
        for idx in stride(from: 0, to: x.count, by: 2) {
            ret.append(a * x[idx])
            ret.append(b * x[idx+1])
        }
        //let a: Double = P[0] * P[1]
        //let b: Double = P[2] * P[3]
        return ret
    }

    // generate x, y pairs from f(x) = 123.45 * x + 987.65
    // with some noise
    let trueP: [Double] = [123.45, 987.65]
    
    var Xs: [Double] = []
    var Ys: [Double] = []
    
    for x in stride(from: 1.0, through: 10.0, by: 1.0) {
        let inx = [x, 1.0]
        let iny = f(P: trueP, x: inx).withNoise(variance: 5)
        Xs.append(inx[0])
        Xs.append(inx[1])
        Ys.append(iny[0])
        Ys.append(iny[1])
        //Xs.append(inx)
        //Ys.append(iny)
    }
    
    // parameters initial value
    //let params: [Double] = [0.0, 0.0]
    let params: [Double] = [100.0, 1000.0]
    
//    for x in stride(from: 1.0, through: 100.0, by: 1.0) {
//        params.append(x)// + Double.random(in: -0.1..<0.1))
//        params.append(1.0)// + Double.random(in: -0.1..<0.1))
//    }

    //let optP = optimizer(f: f, X: outs, P: params)
    let optP = optimizer.optimizeWithInputOutput(f: f, X: Ys, P: params, x: Xs)
    print(optP)
    let ntp = normalize(arr: trueP)
    let nop = normalize(arr: optP)
    
    // calc dist to original params
    let dist: Double = sqrt(zip(ntp, nop).reduce(0.0) { res, val in
        res + (val.0 - val.1) * (val.0 - val.1)
    })
    
    print("dist: \(dist)")
    return dist < 0.1
    
}


func testMatrixFunctions() -> Bool {
    
    let A: [Double] = [
        1.0, 0.0, 0.0,
        -1.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    ]
    
    // test inverse
    let invA = invert(matrix: A, M: 3, N: 3)
    
    var bools: [Bool] = []
    bools.append(invA[1] == 0.0)
    bools.append(invA[3] == 1.0)
    
    // test mul
    
    let LHS: [Double] = [
        1, 4, 7,
        2, 5, 8,
        3, 6, 9
    ]
    
    let RHS: [Double] = [
        10, 11, 12
    ]
    
    let mul = inner(A: LHS, B: RHS, M: 3, P: 3, N: 1)
    
    bools.append(mul[0] == 68.0)
    bools.append(mul[1] == 167.0)
    bools.append(mul[2] == 266.0)
    
    // test trans
    
    let transA = transpose(A: A, M: 3, N: 3)
    bools.append(transA[1] == -1.0)
    
    return bools.allSatisfy { b in
        b
    }
}
