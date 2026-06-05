import SwiftUI
import SceneKit
import ModelIO
import SceneKit.ModelIO
import AppKit

struct SceneKitView: NSViewRepresentable {
    let url: URL
    var autoRotate: Bool

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> SCNView {
        let view = SCNView()
        view.allowsCameraControl = true
        view.autoenablesDefaultLighting = false
        view.antialiasingMode = .multisampling2X
        view.backgroundColor = .clear
        view.preferredFramesPerSecond = 30
        view.defaultCameraController.interactionMode = .orbitTurntable
        return view
    }

    func updateNSView(_ view: SCNView, context: Context) {
        let mod = (try? FileManager.default.attributesOfItem(atPath: url.path))?[.modificationDate] as? Date
        if context.coordinator.url != url || context.coordinator.mod != mod {
            context.coordinator.url = url
            context.coordinator.mod = mod
            let (scene, pivot) = Self.makeScene(url: url)
            view.scene = scene
            context.coordinator.pivotNode = pivot
        }

        if let pivot = context.coordinator.pivotNode {
            if autoRotate && !context.coordinator.isRotating {
                let spin = SCNAction.repeatForever(.rotateBy(x: 0, y: .pi * 2, z: 0, duration: 18))
                spin.timingMode = .linear
                pivot.runAction(spin, forKey: "turntable")
                context.coordinator.isRotating = true
            } else if !autoRotate && context.coordinator.isRotating {
                pivot.removeAction(forKey: "turntable")
                context.coordinator.isRotating = false
            }
        }
    }

    static func makeScene(url: URL) -> (SCNScene, SCNNode?) {
        let scene = SCNScene()
        let container = SCNNode()

        if url.pathExtension.lowercased() == "ply", let (geo, isMesh) = Self.loadPLY(url: url, maxPoints: 200_000) {
            let node = SCNNode(geometry: geo)
            // Point-cloud PLYs are LiTo-native +Z up (reference viewer: viewUp=[0,0,1]);
            // mesh PLYs are already exported +Y up (MeshExtract.writePLY), so no rotation.
            if !isMesh { node.eulerAngles.x = -.pi / 2 }
            container.addChildNode(node)
        } else {
            let asset = MDLAsset(url: url)
            asset.loadTextures()
            let loaded = SCNScene(mdlAsset: asset)
            for child in loaded.rootNode.childNodes { container.addChildNode(child) }
        }

        container.enumerateChildNodes { node, _ in
            guard let geo = node.geometry else { return }
            let isPoints = geo.elements.contains { $0.primitiveType == .point }
            let hasVertexColors = geo.sources.contains { $0.semantic == .color }
            let hasTexture = geo.materials.contains {
                $0.diffuse.contents is NSImage || $0.diffuse.contents is URL
            }
            if isPoints {
                for el in geo.elements {
                    el.pointSize = 5
                    el.minimumPointScreenSpaceRadius = 1
                    el.maximumPointScreenSpaceRadius = 5
                }
                let m = SCNMaterial()
                m.lightingModel = .constant
                m.isDoubleSided = true
                m.diffuse.contents = NSColor.white
                geo.materials = [m]
            } else if hasVertexColors {
                let m = SCNMaterial()
                m.lightingModel = .lambert
                m.isDoubleSided = true
                m.diffuse.contents = NSColor.white
                geo.materials = [m]
            } else if !hasTexture {
                let m = SCNMaterial()
                m.lightingModel = .physicallyBased
                m.diffuse.contents = NSColor(calibratedWhite: 0.82, alpha: 1)
                m.roughness.contents = 0.55
                m.metalness.contents = 0.0
                geo.materials = [m]
            }
        }

        let (minB, maxB) = container.boundingBox
        let center = SCNVector3((minB.x + maxB.x) / 2, (minB.y + maxB.y) / 2, (minB.z + maxB.z) / 2)
        let size = SCNVector3(maxB.x - minB.x, maxB.y - minB.y, maxB.z - minB.z)
        let radius = max(size.x, max(size.y, size.z)) / 2
        container.position = SCNVector3(-center.x, -center.y, -center.z)

        let pivot = SCNNode()
        pivot.addChildNode(container)
        scene.rootNode.addChildNode(pivot)

        let cam = SCNCamera()
        cam.zNear = 0.001
        cam.zFar = Double(max(radius, 0.001)) * 60
        cam.fieldOfView = 35
        let camNode = SCNNode()
        camNode.camera = cam
        let dist: CGFloat = radius <= 0 ? 2 : radius * 3.4
        camNode.position = SCNVector3(0, 0, dist)
        camNode.look(at: SCNVector3Zero)
        scene.rootNode.addChildNode(camNode)

        let key = SCNLight(); key.type = .directional; key.intensity = 850
        let keyNode = SCNNode(); keyNode.light = key
        keyNode.position = SCNVector3(2, 3, 4); keyNode.look(at: SCNVector3Zero)
        scene.rootNode.addChildNode(keyNode)
        let fill = SCNLight(); fill.type = .ambient; fill.intensity = 380
        let fillNode = SCNNode(); fillNode.light = fill
        scene.rootNode.addChildNode(fillNode)

        return (scene, pivot)
    }

    /// Parse a binary little-endian PLY with float x,y,z + uchar red,green,blue per vertex.
    /// Files with an `element face` section come back as a colored triangle mesh
    /// (the splat→mesh export, second tuple element true); pure point clouds subsample
    /// to `maxPoints` for display.
    private static func loadPLY(url: URL, maxPoints: Int = 200_000) -> (SCNGeometry, Bool)? {
        guard let data = try? Data(contentsOf: url) else { return nil }

        guard let headerEnd = data.range(of: Data("end_header\n".utf8)) else { return nil }
        let headerData = data[data.startIndex..<headerEnd.lowerBound]
        guard let header = String(data: headerData, encoding: .utf8) else { return nil }

        var vertexCount = 0
        var faceCount = 0
        var props = [(String, String)]()
        var inVertex = false
        for line in header.components(separatedBy: "\n") {
            let parts = line.split(separator: " ")
            if parts.first == "element" && parts.count >= 3 {
                inVertex = parts[1] == "vertex"
                if inVertex { vertexCount = Int(parts[2]) ?? 0 }
                if parts[1] == "face" { faceCount = Int(parts[2]) ?? 0 }
            } else if parts.first == "property" && inVertex && parts.count >= 3 {
                props.append((String(parts[1]), String(parts[parts.count - 1])))
            }
        }
        guard vertexCount > 0 else { return nil }
        if faceCount > 0 {
            return loadPLYMesh(data: data, bodyStart: headerEnd.upperBound,
                               vertexCount: vertexCount, faceCount: faceCount).map { ($0, true) }
        }

        let bodyStart = headerEnd.upperBound
        let body = data[bodyStart...]

        func typeSize(_ t: String) -> Int {
            switch t {
            case "float", "float32", "int", "int32", "uint", "uint32": return 4
            case "double", "float64": return 8
            case "uchar", "uint8", "char", "int8": return 1
            case "short", "int16", "ushort", "uint16": return 2
            default: return 0
            }
        }
        let stride = props.reduce(0) { $0 + typeSize($1.0) }
        guard body.count >= vertexCount * stride else { return nil }

        var offsets = [String: Int]()
        var off = 0
        for (type, name) in props {
            offsets[name] = off
            off += typeSize(type)
        }
        guard let ox = offsets["x"], let oy = offsets["y"], let oz = offsets["z"] else { return nil }
        let hasColor = offsets["red"] != nil && offsets["green"] != nil && offsets["blue"] != nil

        // Subsample for display if too many points
        let displayCount: Int
        let sampleStride: Int
        if vertexCount > maxPoints {
            sampleStride = vertexCount / maxPoints
            displayCount = maxPoints
        } else {
            sampleStride = 1
            displayCount = vertexCount
        }

        let outStride = hasColor ? MemoryLayout<Float>.size * 6 : MemoryLayout<Float>.size * 3
        var vbuf = Data(count: displayCount * outStride)

        vbuf.withUnsafeMutableBytes { dst in
            body.withUnsafeBytes { src in
                let base = src.baseAddress!
                let out = dst.baseAddress!.assumingMemoryBound(to: Float.self)
                for di in 0..<displayCount {
                    let i = di * sampleStride
                    let row = base + i * stride
                    out[di * (outStride / 4) + 0] = row.advanced(by: ox).loadUnaligned(as: Float.self)
                    out[di * (outStride / 4) + 1] = row.advanced(by: oy).loadUnaligned(as: Float.self)
                    out[di * (outStride / 4) + 2] = row.advanced(by: oz).loadUnaligned(as: Float.self)
                    if hasColor {
                        let or = offsets["red"]!, og = offsets["green"]!, ob = offsets["blue"]!
                        out[di * (outStride / 4) + 3] = Float(row.advanced(by: or).load(as: UInt8.self)) / 255.0
                        out[di * (outStride / 4) + 4] = Float(row.advanced(by: og).load(as: UInt8.self)) / 255.0
                        out[di * (outStride / 4) + 5] = Float(row.advanced(by: ob).load(as: UInt8.self)) / 255.0
                    }
                }
            }
        }

        let posSource = SCNGeometrySource(
            data: vbuf, semantic: .vertex, vectorCount: displayCount,
            usesFloatComponents: true, componentsPerVector: 3,
            bytesPerComponent: 4, dataOffset: 0, dataStride: outStride)

        var sources: [SCNGeometrySource] = [posSource]
        if hasColor {
            sources.append(SCNGeometrySource(
                data: vbuf, semantic: .color, vectorCount: displayCount,
                usesFloatComponents: true, componentsPerVector: 3,
                bytesPerComponent: 4, dataOffset: 12, dataStride: outStride))
        }

        var indices = Data(count: displayCount * 4)
        indices.withUnsafeMutableBytes { ptr in
            let p = ptr.baseAddress!.assumingMemoryBound(to: UInt32.self)
            for i in 0..<displayCount { p[i] = UInt32(i) }
        }
        let element = SCNGeometryElement(
            data: indices, primitiveType: .point,
            primitiveCount: displayCount, bytesPerIndex: 4)
        element.pointSize = 5
        element.minimumPointScreenSpaceRadius = 1
        element.maximumPointScreenSpaceRadius = 5

        let geo = SCNGeometry(sources: sources, elements: [element])
        let mat = SCNMaterial()
        mat.lightingModel = .constant
        mat.isDoubleSided = true
        mat.diffuse.contents = NSColor.white
        geo.materials = [mat]
        return (geo, false)
    }

    /// The splat→mesh PLY: vertices are x,y,z float + r,g,b uchar (15 bytes), faces are
    /// `uchar 3 + 3×int32`. Builds triangle geometry with computed vertex normals so the
    /// lambert shading reads as a surface.
    private static func loadPLYMesh(data: Data, bodyStart: Data.Index,
                                    vertexCount: Int, faceCount: Int) -> SCNGeometry? {
        let vStride = 15
        guard data.count >= bodyStart - data.startIndex + vertexCount * vStride + faceCount * 13 else { return nil }

        var positions = [Float](repeating: 0, count: vertexCount * 3)
        var colors = [Float](repeating: 0, count: vertexCount * 3)
        var indices = [Int32](repeating: 0, count: faceCount * 3)
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            let base = raw.baseAddress!.advanced(by: bodyStart - data.startIndex)
            for v in 0 ..< vertexCount {
                let p = base.advanced(by: v * vStride)
                positions[v * 3] = p.loadUnaligned(as: Float.self)
                positions[v * 3 + 1] = p.advanced(by: 4).loadUnaligned(as: Float.self)
                positions[v * 3 + 2] = p.advanced(by: 8).loadUnaligned(as: Float.self)
                for c in 0 ..< 3 { colors[v * 3 + c] = Float(p.advanced(by: 12 + c).load(as: UInt8.self)) / 255 }
            }
            let faces = base.advanced(by: vertexCount * vStride)
            for f in 0 ..< faceCount {
                let p = faces.advanced(by: f * 13)
                guard p.load(as: UInt8.self) == 3 else { continue }
                for c in 0 ..< 3 { indices[f * 3 + c] = p.advanced(by: 1 + c * 4).loadUnaligned(as: Int32.self) }
            }
        }

        // area-weighted vertex normals
        var normals = [Float](repeating: 0, count: vertexCount * 3)
        for f in 0 ..< faceCount {
            let a = Int(indices[f * 3]), b = Int(indices[f * 3 + 1]), c = Int(indices[f * 3 + 2])
            let ax = positions[a*3], ay = positions[a*3+1], az = positions[a*3+2]
            let ux = positions[b*3] - ax, uy = positions[b*3+1] - ay, uz = positions[b*3+2] - az
            let vx = positions[c*3] - ax, vy = positions[c*3+1] - ay, vz = positions[c*3+2] - az
            let nx = uy*vz - uz*vy, ny = uz*vx - ux*vz, nz = ux*vy - uy*vx
            for i in [a, b, c] { normals[i*3] += nx; normals[i*3+1] += ny; normals[i*3+2] += nz }
        }
        for v in 0 ..< vertexCount {
            let l = (normals[v*3]*normals[v*3] + normals[v*3+1]*normals[v*3+1] + normals[v*3+2]*normals[v*3+2]).squareRoot()
            if l > 1e-12 { normals[v*3] /= l; normals[v*3+1] /= l; normals[v*3+2] /= l }
        }

        func source(_ values: [Float], _ semantic: SCNGeometrySource.Semantic) -> SCNGeometrySource {
            values.withUnsafeBufferPointer {
                SCNGeometrySource(data: Data(buffer: $0), semantic: semantic,
                                  vectorCount: vertexCount, usesFloatComponents: true,
                                  componentsPerVector: 3, bytesPerComponent: 4,
                                  dataOffset: 0, dataStride: 12)
            }
        }
        let element = indices.withUnsafeBufferPointer {
            SCNGeometryElement(data: Data(buffer: $0), primitiveType: .triangles,
                               primitiveCount: faceCount, bytesPerIndex: 4)
        }
        let geo = SCNGeometry(sources: [source(positions, .vertex),
                                        source(normals, .normal),
                                        source(colors, .color)], elements: [element])
        let mat = SCNMaterial()
        mat.lightingModel = .lambert
        mat.isDoubleSided = true
        mat.diffuse.contents = NSColor.white
        geo.materials = [mat]
        return geo
    }

    final class Coordinator {
        var url: URL?
        var mod: Date?
        var pivotNode: SCNNode?
        var isRotating = false
    }
}

// MARK: - Live generation preview (intermediate occupancy dots)

/// White-dot point cloud shown while the DiT is sampling — the intermediate occupancy
/// decodes stream in as flat [x,y,z]* world coords (LiTo z-up, ≈[-1,1]) and visibly
/// coalesce into the final shape. Geometry swaps in place; the slow turntable keeps
/// spinning across updates.
struct LiveCloudView: NSViewRepresentable {
    let points: [Float]              // count·3, z-up world
    let generation: Int              // bump to force a geometry refresh

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> SCNView {
        let view = SCNView()
        view.allowsCameraControl = false
        view.backgroundColor = .clear
        view.autoenablesDefaultLighting = false
        view.preferredFramesPerSecond = 30

        let scene = SCNScene()
        let pivot = SCNNode()
        let cloud = SCNNode()
        cloud.eulerAngles.x = -.pi / 2          // LiTo z-up → SceneKit y-up
        pivot.addChildNode(cloud)
        scene.rootNode.addChildNode(pivot)
        let spin = SCNAction.repeatForever(.rotateBy(x: 0, y: .pi * 2, z: 0, duration: 14))
        pivot.runAction(spin)

        let cam = SCNCamera()
        cam.zNear = 0.01; cam.zFar = 50; cam.fieldOfView = 35
        let camNode = SCNNode()
        camNode.camera = cam
        camNode.position = SCNVector3(0, 0.25, 3.6)
        camNode.look(at: SCNVector3Zero)
        scene.rootNode.addChildNode(camNode)

        view.scene = scene
        context.coordinator.cloudNode = cloud
        return view
    }

    func updateNSView(_ view: SCNView, context: Context) {
        guard context.coordinator.generation != generation,
              let cloud = context.coordinator.cloudNode, !points.isEmpty else { return }
        context.coordinator.generation = generation
        cloud.geometry = Self.dotGeometry(points)
    }

    private static func dotGeometry(_ pts: [Float]) -> SCNGeometry {
        let count = pts.count / 3
        let source = pts.withUnsafeBufferPointer {
            SCNGeometrySource(data: Data(buffer: $0), semantic: .vertex, vectorCount: count,
                              usesFloatComponents: true, componentsPerVector: 3,
                              bytesPerComponent: 4, dataOffset: 0, dataStride: 12)
        }
        var indices = [UInt32](repeating: 0, count: count)
        for i in 0 ..< count { indices[i] = UInt32(i) }
        let element = indices.withUnsafeBufferPointer {
            SCNGeometryElement(data: Data(buffer: $0), primitiveType: .point,
                               primitiveCount: count, bytesPerIndex: 4)
        }
        element.pointSize = 3
        element.minimumPointScreenSpaceRadius = 1
        element.maximumPointScreenSpaceRadius = 3
        let geo = SCNGeometry(sources: [source], elements: [element])
        let m = SCNMaterial()
        m.lightingModel = .constant
        m.diffuse.contents = NSColor.white
        m.transparency = 0.92
        geo.materials = [m]
        return geo
    }

    final class Coordinator {
        var cloudNode: SCNNode?
        var generation = -1
    }
}
