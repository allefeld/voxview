#define GAMMA
//#define AA
#define DEBUG


// TODO: fix weird rendering artefact, especially visible for single voxel

// volume data
uniform sampler3D vol;

// volume coordinate system (affine transformation
uniform mat3 AM;
uniform vec3 AO;
uniform mat3 AiM;

// value at which the volume is to be thresholded
uniform float threshold;

// how the volume appears: coefficients for Phong illumination
//   see http://devernay.free.fr/cours/opengl/materials.html
// "brass"
const vec3 ka = vec3(0.329412, 0.223529, 0.027451);
const vec3 kd = vec3(0.780392, 0.568627, 0.113725);
const vec3 ks = vec3(0.992157, 0.941176, 0.807843);
const float alpha = 27.89744;
//// "black rubber"
//const vec3 ka = vec3(0.02, 0.02, 0.02);
//const vec3 kd = vec3(0.01, 0.01, 0.01);
//const vec3 ks = vec3(0.4, 0.4, 0.4);
//const float alpha = 10.;
//// "jade"
//const vec3 ka = vec3(0.135, 0.2225, 0.1575);
//const vec3 kd = vec3(0.54, 0.89, 0.63);
//const vec3 ks = vec3(0.316228, 0.316228, 0.316228);
//const float alpha = 12.8;


// direction of light
const vec3 lightDir = normalize(vec3(-2., 1., 1.));


// numerical infinity
const float inf = 1. / 0.;
// numerical not a number
const float nan = 0. / 0;


// background of scene
// in
//   vec3 ray:  view direction
// returns
//   vec3:      color
vec3 background(in vec3 ray) {
    // light
    if (dot(lightDir, ray) >= cos(radians(0.5))) {
        return vec3(1., 1., 1.);
    }
    // black to green-blue in z-direction
    return (1. + ray.z) / 2. * vec3(0., 0.5, 1.);
}


// cube tracing algorithm – trilinear
//   Processes a line segment from the grid tracing algorithm, to find the point
// at which the ray intersects with a surface defined within each grid cube.
// The surface is defined by trilinear interpolation crossing `threshold`.
// in
//   vec3 pos:      entry point into grid cube (can be within)
//   vec3 posNext:  exit point from the grid cube
// out
//   vec3 p:        intersection point of ray and surface
//   vec3 n:        normal vector of the surface at the intersection point
// returns
//   bool:          whether an intersection point was found
bool cubetracer(in vec3 pos, in vec3 posNext,
                  out vec3 p, out vec3 n) {
    int i, j, k;
    float v[2][2][2];
    // the lower vertex of the grid cube the line segment crosses
    ivec3 lower = ivec3(floor((pos + posNext) / 2.));
    // extract data from surrounding vertices
    float vMax = -inf;
    float vMin = inf;
    for (i = 0; i < 2; i++) {
        for (j = 0; j < 2; j++) {
            for (k = 0; k < 2; k++) {
                v[i][j][k] = texelFetch(vol, lower + ivec3(i, j, k), 0).r;
                if (v[i][j][k] > vMax) { vMax = v[i][j][k]; }
                if (v[i][j][k] < vMin) { vMin = v[i][j][k]; }
            }
        }
    }
    // quick check whether there can be a threshold crossing
    if ((vMax < threshold) || (vMin > threshold)) {
        return false;
    }
    // -> determine threshold crossing
    // coefficients for line through grid cube, p = a + b t
    vec3 b = posNext - pos;
    vec3 a = pos - lower;
    // coefficients of trilinear interpolation within grid cube
    float c     = + v[0][0][0];
    float cx    = - v[0][0][0] + v[1][0][0];
    float cy    = - v[0][0][0] + v[0][1][0];
    float cz    = - v[0][0][0] + v[0][0][1];
    float cxy   = + v[0][0][0] - v[0][1][0] - v[1][0][0] + v[1][1][0];
    float cxz   = + v[0][0][0] - v[0][0][1] - v[1][0][0] + v[1][0][1];
    float cyz   = + v[0][0][0] - v[0][0][1] - v[0][1][0] + v[0][1][1];
    float cxyz  = - v[0][0][0] + v[0][0][1] + v[0][1][0] - v[0][1][1]
                  + v[1][0][0] - v[1][0][1] - v[1][1][0] + v[1][1][1];
    // & along line through grid cube
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
    // method using extrema, adapted from
    //   Marmitt et al. Fast and accurate ray-voxel intersection techniques for
    // iso-surface ray tracing. In Vision, Modeling, and Visualization 2004.
    // http://hodad.bioen.utah.edu/~wald/Publications/2004/iso/IsoIsec_VMV2004.pdf
    // "Algorithm 3" -----------------------------------------------------------
    float t0 = 0.;
    float f0 = d0;                  // d0 + (d1 + (d2 + d3 * t0) * t0) * t0;
    float t1 = 1.;
    float f1 = d0 + d1 + d2 + d3;   // d0 + (d1 + (d2 + d3 * t1) * t1) * t1;
    // Find extrema by looking at f'(t) = d1 + 2 d2 t + 3 d3 t²
    // solutions are t = (-d2 ± sqrt(d2² - 3 d1 d3)) / (3 d3)
    if ((d2 * d2 > 3 * d1 * d3) && (d3 != 0)) {
        // f' has real (and finite) roots
        // tm = smaller root of f'
        float tm = (-d2 - sign(d3) * sqrt(d2 * d2 - 3 * d1 * d3)) / (3 * d3);
        if ((t0 < tm) && (tm < t1)) {
            // if tm splits the interval
            // examine [t0 tm] for a threshold crossing
            float fm = d0 + (d1 + (d2 + d3 * tm) * tm) * tm;
            if (sign(fm - threshold) == sign(f0 - threshold)) {
                // no threshold crossing -> advance to [tm t1]
                t0 = tm;
                f0 = fm;

                // tp = second root of f'
                float tp = (-d2 + sign(d3) * sqrt(d2 * d2 - 3 * d1 * d3)) / (3 * d3);
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
            } else {
                // threshold crossing -> calculate it in [t0 tm]
                t1 = tm;
                f1 = fm;
            }
        }
    }
    // final check whether a threshold crossing segment has been found
    if (sign(f0 - threshold) == sign(f1 - threshold)) {
        return false;
    }
    // now we can be sure that the segment [t0 t1] contains a threshold crossing
    // calculate it via repeated linear interpolation
    const int N = 1;
    // approximate threshold
    // f = f0 + (f1 - f0) * (t - t0) / (t1 - t0)  =  threshold
    float t = t0 + (t1 - t0) * (threshold - f0) / (f1 - f0);
    for (int i = 1; i <= N; i++) {
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
    // position at which the threshold crossing occurs
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
    p += lower;
    return true;
    // end of "Algorithm 3" ----------------------------------------------------
}


// grid tracing algorithm
//   The algorithm produces a sequence of points on the dir, in order of
// increasing distance from the start, such that each pair of subsequent points
// defines a line segment that lies within a "grid cube", defined as the space
// between neighboring voxel centers.
//   The line segments are then processed by `cubetracer` to find the point at
// which the ray intersects with a surface defined within each grid cube.
// in
//   int id:        which volume to traverse            TODO
//   vec3 start:    from where
//   vec3 dir:      in which direction (normalized!)
// out
//   float d:       at what distance
//                  The value 0 indicates that start is within the surface,
//                  the value inf that no surface was encountered. In that case,
//                  n is undefined.
//   vec3 n:        normal vector of the surface
void gridtracer(in int id, in vec3 start, in vec3 dir,
                out float d, out vec3 n) {
    // extent of volume
    vec3 shape = textureSize(vol, 0);
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
            n = vec3(nan, nan, nan);
            return;
        }
    }

    // produce new points sequentially
    int l = 0;
    while (all(greaterThanEqual(pos, vec3(-1., -1, -1.))
                    && lessThanEqual(pos, shape))) {
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
        if (cubetracer(pos, posNext, p, n)) {
            d = dot((p - start), dir);
            return;
        }

        // The next point becomes the new start point.
        pos = posNext;

        // just in case, prevent an infinite loop
        l++;    // number of line segments so far
        // more than there can possibly be?
        if (l > shape[0] + shape[1] + shape[2] + 3) {
            error = vec4(0., 1., 0., 0.);
            d = distMin;
            n = vec3(nan, nan, nan);
            return;
        }
    }

    // nothing found until the edge of the extended grid
    d = inf;
    n = vec3(nan, nan, nan);
    return;
}

// main function
// out
//   vec4 fragColor:    color the current fragment (pixel) should be display in
// in
//   vec2 fragCoord:    coordinates of the current (pixel)
//                      values are in [0.5, resolution - 0.5]
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    // translate pixel indices into coordinate system where (0, 0) is
    // the center, and the frame is fit into the range (-1, -1) to (1, 1)
    vec2 coord = (2. * fragCoord - resolution) /
        max(resolution.x, resolution.y);

    // ray from the camera through the pixel
    vec3 ray = normalize(camMatrix * vec3(coord, 1.0));

    // trace the volume grid
    // TODO: multiple volumes
    //   transform camPos & ray from world into voxel space
    vec3 vcp = AiM * (camPos - AO);
    vec3 vr = AiM * ray;
    // find intersection of ray with surface in voxel space
    float vd;   // for voxel-space distance
    vec3 vn;    // for voxel-space normal vector
    gridtracer(1, vcp, vr, vd, vn);
    //   position of intersection in voxel space
    vec3 vp = vcp + vd * vr;
    //   transform position from voxel into world space
    vec3 p = AM * vp + AO;
    //   transform normal vector from voxel into world space
    //     The normal vector (derived from the gradient) is an element of the
    //   dual space and therefore transforms covariantly. The reason is that
    //   orthogonality must be preserved:
    //     n' v = 0, w = M v, m = N n such that m' w = 0
    //     0 = m' w = n' N' M v in general
    //     ->  N' M = I  ->  N = inv(M)'
    vec3 n = transpose(AiM) * vn;

    // special cases
    // TODO how do we want to deal with inside position
    // through a mirror, darkly?
    // with the current cubetracing we cannot even detect it this way
//    if (d == 0) {
//        // inside -> paint it gray
//        fragColor = vec4(0.3, 0.3, 0.3, 1.);
//        return;
//    }
    if (vd == inf) {
        // not found -> background
        fragColor = vec4(background(ray), 1.);
        return;
    }

    // Phong illumination
    //   see https://en.wikipedia.org/wiki/Phong_reflection_model#Description
    // TODO: multiple lights? – methinks at least two
    // normalized vectors
    vec3 v = normalize(camPos - p);        // direction to the viewer
    vec3 l = lightDir;                      // direction to the light
    vec3 r = normalize(reflect(-l, n));     // direction of reflected light
    // scalar products
    float drv = dot(r, v);
    float dln = dot(l, n);
    // sign constraints
    drv = drv * step(0., drv) * step(0., dln);
    dln = dln * step(0., dln);
    // illumination: ambient + diffuse + specular
    vec3 k = ka + kd * dln + ks * pow(drv, alpha);

    // prevent color overflow
    // TODO: this is just a hack
    // We would have to compute the factor across all materials.
    vec3 ksum = ka + kd + ks;
    float factor = 1. / max(max(ksum.r, ksum.g), ksum.b);

    fragColor = vec4(factor * k, 1.);
}



// Reference:
// - GLSL
//   https://www.khronos.org/opengl/wiki/Core_Language_%28GLSL%29
//   https://www.khronos.org/registry/OpenGL-Refpages/gl4/index.php
// - distance functions
//   http://iquilezles.org/www/articles/distfunctions/distfunctions.htm
//   http://9bitscience.blogspot.com/2013/07/raymarching-distance-fields_14.html
//   http://iquilezles.org/www/articles/smin/smin.htm
