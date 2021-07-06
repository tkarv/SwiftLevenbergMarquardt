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

func lmSolvesNonlinearEquation() -> Bool {
    // TODO: add nonlinear test
    return true
}

func lmSolvesLinearEquation() -> Bool {
    // prepare function to calculate
    // f(x) = p_0 * x_0 + p_1 * x_1
    func f(P: [Double], X: [Double]) -> [Double] {
        let a: Double = P[0] * X[0]
        let b: Double = P[1] * X[1]
        return [a, b]
    }

    // generate x, y pairs from f(x) = 123.45 * x + 987.65
    // with some noise
    let trueP: [Double] = [123.45, 987.65]

    var Xs: [[Double]] = []
    var Ys: [[Double]] = []

    for x in stride(from: 1.0, through: 100.0, by: 1.0) {
        let inx = [x, 1.0]
        let iny = f(P: trueP, X: inx).withNoise(variance: 100)
        Xs.append(inx)
        Ys.append(iny)
    }

    // parameters initial value
    let params: [Double] = [0.0, 0.0]

    let optP = levenbergMarquardt(f: f, X: Ys, P: params, x: Xs)

    let ntp = normalize(arr: trueP)
    let nop = normalize(arr: optP)
    
    // calc dist to original params
    let dist: Double = sqrt(zip(ntp, nop).reduce(0.0) { res, val in
        res + (val.0 - val.1) * (val.0 - val.1)
    })
    
    print("dist: \(dist)")
    return sqrt(dist) < 0.1
    
}

