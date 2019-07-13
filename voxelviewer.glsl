#version 130

// for debugging, see main()
vec4 error = vec4(0., 0., 0., 0.);

#define NV 0                // 0 replaced by len(self.volumes)
#define NS 0                // 0 replaced by len(self.surfaces)

#define sdf sdfSpheroid
#define cubetracer cubetracerST

uniform vec2  resolution;   // viewport resolution (in pixels)
uniform float time;         // shader playback time (in seconds)
uniform vec3  camPos;       // camera position
uniform mat3  camMatrix;    // camera matrix


// volumes
struct Volume {
    // volume data
    sampler3D data;
    // volume coordinate system (affine transformation)
    mat3 AM;        // matrix voxel -> world
    vec3 AO;        // offset voxel -> world
    mat3 AiM;       // matrix world -> voxel
};
uniform Volume vol[NV];


// surfaces
struct Surface {
    // volume to be thresholded
    int volID;
    // threshold value
    float threshold;
    // appearance: Phong illumination coefficients
    vec3 ka;        // ambient
    vec3 kd;        // diffuse
    vec3 ks;        // specular
    float alpha;    // shininess
};
uniform Surface surf[NS];


// direction of lights
//   Four lights in tetrahedral configuration reach everywhere.
const vec3 lightDir[4] = vec3[4](
    vec3(1., 1., 1.),
    vec3(1., -1., -1.),
    vec3(-1., 1., 1.),
    vec3(1., 1., -1.));


// numerical infinity
const float inf = 1. / 0.;


// background of scene
// in
//   vec3 ray:  view direction
// returns
//   vec3:      color
vec3 background(in vec3 ray) {
    // black to green-blue in z-direction
    return (1. + ray.z) / 2. * vec3(0., 0.5, 1.);
}


// cube tracing algorithm – trilinear
//   Processes a line segment from the grid tracing algorithm, to find the point
// at which the ray intersects with a surface defined within each grid cube.
// The surface is defined by trilinear interpolation crossing the threshold.
//   Accepts and returns coordinates in voxel space.
// in
//   int volID          ID of volume
//   float threshold    threshold defining the surface
//   vec3 pos:          entry point into grid cube (can be within)
//   vec3 posNext:      exit point from the grid cube
// out
//   vec3 p:            intersection point of ray and surface
//   vec3 n:            normal vector of the surface at the intersection point
// returns
//   bool:              whether an intersection point was found
bool cubetracerTL(in int volID, in float threshold, in vec3 pos, in vec3 posNext,
                  out vec3 p, out vec3 n) {
    // the lower vertex of the grid cube the line segment crosses
    ivec3 lower = ivec3(floor((pos + posNext) / 2.));
    // extract data from surrounding vertices
    float v[8];
    float vMax = -inf;
    float vMin = inf;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                // emulate three-dimensional array
                int index = i * 4 + j * 2 + k;
                // get voxel data from volume
                v[index] = texelFetch(vol[volID].data,
                    lower + ivec3(i, j, k), 0).r;
                if (v[index] > vMax) { vMax = v[index]; }
                if (v[index] < vMin) { vMin = v[index]; }
            }
        }
    }
    // quick check whether there can be a threshold crossing at all
    if (sign(vMin - threshold) == sign(vMax - threshold)) {
        return false;
    }
    // coefficients for line segment through grid cube, p = a + b t, t in [0, 1]
    // coordinates translated to unit cube
    vec3 b = posNext - pos;
    vec3 a = pos - lower;
    // coefficients of trilinear interpolation within unit cube, f(x, y, z)
    // x, y, z in [0, 1]
    float c     = + v[0];
    float cx    = - v[0] + v[4];
    float cy    = - v[0] + v[2];
    float cz    = - v[0] + v[1];
    float cxy   = + v[0] - v[2] - v[4] + v[6];
    float cxz   = + v[0] - v[1] - v[4] + v[5];
    float cyz   = + v[0] - v[1] - v[2] + v[3];
    float cxyz  = - v[0] + v[1] + v[2] - v[3] + v[4] - v[5] - v[6] + v[7];
    // coefficients of trilinear interpolation along line through unit cube
    // f(t) = d0 + d1 t + d2 t² + d3 t³
    float d0 = c + cx * a.x + cy * a.y + cz * a.z
             + cxy * a.x * a.y + cxz * a.x * a.z + cyz * a.y * a.z
             + cxyz * a.x * a.y * a.z;
    float d1 = cx * b.x + cy * b.y + cz * b.z
             + cxy * (a.x * b.y + b.x * a.y)
             + cxz * (a.x * b.z + b.x * a.z)
             + cyz * (a.y * b.z + b.y * a.z)
             + cxyz * (a.x * b.y * a.z + a.y * b.x * a.z + a.x * a.y * b.z);
    float d2 = cxy * b.x * b.y + cxz * b.x * b.z + cyz * b.y * b.z
             + cxyz * (a.x * b.y * b.z + a.y * b.x * b.z + b.x * b.y * a.z);
    float d3 = cxyz * b.x * b.y * b.z;
    // search for threshold crossing along the line segment
    // via method using extrema: "Algorithm 3" in
    //   Marmitt et al. Fast and accurate ray-voxel intersection techniques for
    // iso-surface ray tracing. In Vision, Modeling, and Visualization 2004.
    // http://hodad.bioen.utah.edu/~wald/Publications/2004/iso/IsoIsec_VMV2004.pdf
    //
    // initial segment is [0, 1]
    float t0 = 0.;
    float f0 = d0;                  // d0 + (d1 + (d2 + d3 * t0) * t0) * t0;
    float t1 = 1.;
    float f1 = d0 + d1 + d2 + d3;   // d0 + (d1 + (d2 + d3 * t1) * t1) * t1;
    // Find extrema by looking at f'(t) = d1 + 2 d2 t + 3 d3 t²
    // solutions are t = -d2 / (3 d3) ± sqrt(d2² - 3 d1 d3) / (3 d3)
    if ((d2 * d2 > 3 * d1 * d3) && (d3 != 0)) {
        // f' has two real (and finite) roots
        float pm = abs(sqrt(d2 * d2 - 3 * d1 * d3) / (3 * d3));
        // tm = smaller root of f'
        float tm = -d2 / (3 * d3) - pm;
        if ((t0 < tm) && (tm < t1)) {
            // if tm splits the interval
            // examine [t0 tm] for a threshold crossing
            float fm = d0 + (d1 + (d2 + d3 * tm) * tm) * tm;
            if (sign(fm - threshold) == sign(f0 - threshold)) {
                // no threshold crossing -> advance to [tm t1]
                t0 = tm;
                f0 = fm;
            } else {
                // threshold crossing -> calculate it in [t0 tm]
                t1 = tm;
                f1 = fm;
            }
        }
        // tp = second root of f'
        float tp = -d2 / (3 * d3) + pm;
        if ((t0 < tp) && (tp < t1)) {
            // if tp splits the interval
            // examine [t0 tp] for a threshold crossing
            float fp = d0 + (d1 + (d2 + d3 * tp) * tp) * tp;
            if (sign(fp - threshold) == sign(f0 - threshold)) {
                // no threshold crossing -> advance to [tp t1]
                t0 = tp;
                f0 = fp;
            } else {
                // threshold crossing -> look for it in [t0, tp]
                t1 = tp;
                f1 = fp;
            }
        }
    }
    // final check whether a threshold crossing segment has been found
    if (sign(f0 - threshold) == sign(f1 - threshold)) {
        return false;
    }
    // now we can be sure that the segment [t0 t1] contains a threshold crossing
    // approximate it via repeated linear interpolation
    // f = f0 + (f1 - f0) * (t - t0) / (t1 - t0)  =  threshold
    float t = t0 + (t1 - t0) * (threshold - f0) / (f1 - f0);
    for (int i = 1; i <= 2; i++) {  // 2 iterations are generally sufficient
        float f = d0 + (d1 + (d2 + d3 * t) * t) * t;
        if (sign(f - threshold) == sign(f0 - threshold)) {
            // approximation too low -> look for it in [t, t1]
            t0 = t;
            f0 = f;
        } else {
            // approximation too high -> look for it in [t0, t]
            t1 = t;
            f1 = f;
        }
        // new interpolation
        t = t0 + (t1 - t0) * (threshold - f0) / (f1 - f0);
    }
    // position within unit cube at which the threshold crossing occurs
    p = a + b * t;
    // trilinear gradient at position
    vec3 grad = vec3(cx + cxy * p.y + cxz * p.z + cxyz * p.y * p.z,
                     cy + cxy * p.x + cyz * p.z + cxyz * p.x * p.z,
                     cz + cxz * p.x + cyz * p.y + cxyz * p.x * p.y);
    // surface normal
    if (d0 < threshold) {
        // looking at surface from outside
        n = -normalize(grad);
    } else {
        // looking at surface from inside
        n = normalize(grad);
    }
    // re-translate coordinates to voxel space
    p += lower;
    return true;
}


// signed distance function for spheres placed at voxel positions
// with above-threshold values (8 vertices of grid cube)
// in
//   vec3 p             position within unit cube
//   float v[8]         voxel data at surrounding vertices
//   float threshold    threshold selecting voxels
// out
//   float d            signed distance of p
//   vec3 n             normalized gradient of signed distance at p
void sdfSpheroid(in vec3 p, in float v[8], in float threshold,
                out float d, out vec3 n) {
    d = inf;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                // emulate three-dimensional array
                int index = i * 4 + j * 2 + k;
                // voxel data above-threshold?
                if (v[index] >= threshold) {
                    // distance from sphere at voxel center with radius 0.5
                    float di = length(p - vec3(i, j, k)) - 0.5;
                    // form minimum across vertices to form sdf union
                    if (di < d) {
                        d = di;
                        // also record normalized gradient
                        n = normalize(p - vec3(i, j, k));
                    }
                }
            }
        }
    }
}

float sdfCuboid(vec3 p, float v[8], float threshold) {
    float dMin = inf;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                int index = i * 4 + j * 2 + k;
                if (v[index] >= threshold) {
                    vec3 D = abs(p - vec3(i, j, k)) - 0.5;
                    float d = length(max(D, 0.0));
                    dMin = min(dMin, d);
                }
            }
        }
    }
    return dMin;
}

// cube tracing algorithm – sphere tracing of signed distance function
//   Processes a line segment from the grid tracing algorithm, to find the point
// at which the ray intersects with a surface defined within each grid cube.
// The surface is defined by a signed distance function.
//   Accepts and returns coordinates in voxel space.
// in
//   int volID          ID of volume
//   float threshold    threshold defining the surface
//   vec3 pos:          entry point into grid cube (can be within)
//   vec3 posNext:      exit point from the grid cube
// out
//   vec3 p:            intersection point of ray and surface
//   vec3 n:            normal vector of the surface at the intersection point
// returns
//   bool:              whether an intersection point was found
bool cubetracerST(in int volID, in float threshold, in vec3 pos, in vec3 posNext,
                  out vec3 p, out vec3 n) {
    // the lower vertex of the grid cube the line segment crosses
    ivec3 lower = ivec3(floor((pos + posNext) / 2.));
    // extract data from surrounding vertices
    float v[8];
    float vMax = -inf;
    float vMin = inf;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                // emulate three-dimensional array
                int index = i * 4 + j * 2 + k;
                // get voxel data from volume
                v[index] = texelFetch(vol[volID].data,
                    lower + ivec3(i, j, k), 0).r;
                if (v[index] > vMax) { vMax = v[index]; }
                if (v[index] < vMin) { vMin = v[index]; }
            }
        }
    }
    // quick check whether there is an above-threshold voxel value at all
    if (vMax < threshold) {
        return false;
    }
    // coefficients for line segment through grid cube, p = a + b t, t in [0, 1]
    // coordinates translated to unit cube
    vec3 b = posNext - pos;
    vec3 a = pos - lower;
    // sphere tracing
    float t = 0.;
    int steps = 0;
    while ((t <= 1.) && (steps < 100)) {
        // compute position
        p = a + b * t;
        // compute signed distance function
        float d;
        sdf(p, v, threshold, d, n);
        // if (almost) inside, surface found
        if (d <= 1e-4) {
            // re-translate coordinates to voxel space
            p += lower;
            return true;
        }
        // else, march along the line
        t += d / length(b);
        steps += 1;
    }
    return false;
}


// grid tracing algorithm
//   The algorithm produces a sequence of points on the dir, in order of
// increasing distance from the start, such that each pair of subsequent points
// defines a line segment that lies within a "grid cube", defined as the space
// between neighboring voxel centers.
//   The line segments are then processed by `cubetracer` to find the point at
// which the ray intersects with a surface defined within each grid cube.
// in
//   int volID          ID of volume
//   float threshold    threshold defining the surface
//   vec3 start:        from where
//   vec3 dir:          in which direction (normalized!)
// out
//   float d:           at what distance
//                        The value inf indicates that no surface was
//                      encountered. In that case, n is undefined.
//   vec3 n:            normal vector of the surface
void gridtracer(in int volID, in float threshold, in vec3 start, in vec3 dir,
                out float d, out vec3 n) {
    // extent of volume
    vec3 shape = textureSize(vol[volID].data, 0);
    // 1) Get the first point.
    vec3 pos;
    // Check whether the start position lies within the extended grid.
    if (all(greaterThanEqual(start, vec3(-1., -1., -1.))
            && lessThanEqual(start, shape))) {
        // If yes, the start position is the first point.
        pos = start;
    } else {
        // If not, the first point is the first intersection of the direction
        // with the edge of the extended grid (at -1, shape). We go through all
        // candidates, filter them, and pick the one closest to the start.
        float distMin = inf;
        // extended grid edges along dimensions
        for (int a = 0; a < 3; a++) {
            // lower / upper edge
            for (int i = 0; i < 2; i++) {
                int val = int(i == 0 ? -1 : shape[a]);
                // candidate distance
                float dist = (val - start[a]) / dir[a];
                // Check whether the candidate is at nonnegative distance,
                // and closer to the start than the previous candidates.
                if ((dist <= 0) || (dist >= distMin)) continue;
                // candidate position
                vec3 p = start + dist * dir;
                // fix imprecisely recovered integer coordinate
                p[a] = val;
                // Check whether the candidate lies within the extended grid
                // (in all dimensions).
                if (! all(greaterThanEqual(p, vec3(-1., -1, -1.))
                    && lessThanEqual(p, shape))) continue;
                // If all checks were passed, remember candidate.
                distMin = dist;
                pos = p;
            }
        }
        // If none of the candidates have been suitable,
        // there is no intersection and the ray misses the extended grid.
        if (distMin == inf) {
            d = inf;
            n = vec3(0, 0, 0);
            return;
        }
    }
    // produce new points sequentially
    int l = 0;
    while (all(greaterThanEqual(pos, vec3(-1., -1, -1.))
                    && lessThanEqual(pos, shape))) {
        // FIXME this allows the last line segment to be beyond the extended grid
        // 2) Get the next point.
        vec3 posNext;
        // Skip a little bit ahead, to avoid getting stuck.
        // Not entirely clear why this is necessary.
        vec3 posA = pos + dir * 1e-4;
        // We go through all candidates, filter them, and pick the closest one.
        float distMin = inf;
        // grid cube edges along dimensions
        for (int a = 0; a < 3; a++) {
            // Depending on the direction, go to next larger / smaller integer.
            int val = int(dir[a] > 0 ? floor(posA[a]) + 1 : ceil(posA[a]) - 1);
            // candidate distance
            float dist = (val - start[a]) / dir[a];
            // Check whether the candidate is closer.
            if (dist >= distMin) continue;
            // candidate position
            vec3 p = start + dist * dir;
            // fix imprecisely recovered integer coordinate
            p[a] = val;
            // Remember candidate
            distMin = dist;
            posNext = p;
        }
        // 3) Process the line segment.
        vec3 p;
        if (cubetracer(volID, threshold, pos, posNext,
                p, n)) {
            d = dot((p - start), dir);
            return;
        }
        // The next point becomes the new start point.
        pos = posNext;
        // just in case, prevent an infinite loop
        l++;            // number of line segments so far
        // more than there can possibly be?
        if (l > shape[0] + shape[1] + shape[2] + 3) {
            error = vec4(0., 1., 0., 0.);
            d = distMin;
            n = vec3(0, 0, 0);
            return;
        }
    }
    // nothing found until the edge of the extended grid
    d = inf;
    n = vec3(0, 0, 0);
    return;
}


// main function
// in
//   vec2 fragCoord:    coordinates of the current (pixel)
//                      values are in [0.5, resolution - 0.5]
// out
//   vec4 fragColor:    color the current fragment (pixel) should be shown in
void mainImage(in vec2 fragCoord,
               out vec4 fragColor) {
    // translate pixel indices into coordinate system where (0, 0) is
    // the center, and the frame is fit into the range (-1, -1) to (1, 1)
    vec2 coord = (2. * fragCoord - resolution) /
        max(resolution.x, resolution.y);

    // ray from the camera through the pixel
    vec3 ray = normalize(camMatrix * vec3(coord, 1.0));

    // trace the volume grid
    float dMin = inf;
    vec3 p, n;
    int surfID;
    for (int sid = 0; sid < NS; sid++) {
        // get volume underlying surface
        int volID = surf[sid].volID;
        // transform camPos from world into voxel space
        vec3 vcp = vol[volID].AiM * (camPos - vol[volID].AO);
        // transform ray from world into voxel space
        vec3 vr = vol[volID].AiM * ray;
        vec3 vdir = normalize(vr);      // as a normalized vector
        // find intersection of ray with surface in voxel space
        float vd;   // for voxel-space distance
        vec3 vn;    // for voxel-space normal vector
        gridtracer(volID, surf[sid].threshold, vcp, vdir, vd, vn);
        // world-space distance
        float d = vd / length(vr);
        // if new distance is smaller
        if (d < dMin) {
            // record new distance
            dMin = d;
            // position in world space
            p = camPos + d * ray;
            // transform normal vector from voxel into world space
            //   The normal vector (derived from the gradient) is an element of the
            // dual space and therefore transforms covariantly. The reason is that
            // orthogonality must be preserved:
            //   n' v = 0, w = M v, m = N n, N such that m' w = 0
            //   0 = m' w = n' N' M v in general
            //   ->  N' M = I  ->  N = inv(M)'
            n = normalize(transpose(vol[volID].AiM) * vn);
            // record surface ID
            surfID = sid;
        }
    }

    // special case
    if (dMin == inf) {
        // not found -> background
        fragColor = vec4(background(ray), 1.);
        return;
    }

    // Phong illumination
    //   see https://en.wikipedia.org/wiki/Phong_reflection_model#Description
    // ambient illumination
    vec3 k = surf[surfID].ka;
    // normalized vectors
    vec3 v = normalize(camPos - p);             // direction to the viewer
    for (int i = 0; i < lightDir.length; i++) {
        vec3 l = normalize(lightDir[i]);        // direction to light
        vec3 r = normalize(reflect(-l, n));     // direction of reflected light
        // scalar products
        float drv = dot(r, v);
        float dln = dot(l, n);
        // sign constraints
        drv = drv * step(0., drv) * step(0., dln);
        dln = dln * step(0., dln);
        // + diffuse illumination + specular illumination
        k += surf[surfID].kd * dln
           + surf[surfID].ks * pow(drv, surf[surfID].alpha);
    }

    fragColor = vec4(k, 1.);
}


// footer ----------------------------------------------------------------------

// this function wraps error checking and gamma correction
out vec4 fragColor;
void main() {
    mainImage(gl_FragCoord.xy,
        fragColor);

    // error checking
    // If there are out-of-range color components, show it as error color.
    if (clamp(fragColor, 0., 1.) != fragColor) {
        error = vec4(clamp(fragColor, 0., 1.).xyz, 0.);
    }
    // If an error color has been set, display it.
    // Alpha component is used to induce blinking.
    if (error != vec4(0., 0., 0., 0.)) {
        if (fract(time * 2.) < 0.5) {
            fragColor = error;
        } else {
            fragColor = error * error.a;
        }
        return; // skip gamma correction
    }
    // conversion from linear to sRGB colorspace ("gamma correction")
    // adapted from https://www.shadertoy.com/view/lscSzl
    vec3 linearRGB = fragColor.rgb;
    vec3 a = 12.92 * linearRGB;
    vec3 b = 1.055 * pow(linearRGB, vec3(1.0 / 2.4)) - 0.055;
    vec3 c = step(vec3(0.0031308), linearRGB);
    fragColor = vec4(mix(a, b, c), fragColor.a);
}
