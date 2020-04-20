#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>
#include <GL/freeglut.h>
#endif

const unsigned int windowWidth = 600, windowHeight = 600;

int majorVersion = 3, minorVersion = 3;

struct vec3
{
    float x, y, z;
    vec3(float x0 = 0, float y0 = 0, float z0 = 0)
    {
        x = x0;
        y = y0;
        z = z0;
    }
    vec3 operator*(float a) const { return vec3(x * a, y * a, z * a); }
    vec3 operator/(float d) const { return vec3(x / d, y / d, z / d); }
    vec3 operator/(const vec3 &d) const { return vec3(x / d.x, y / d.y, z / d.z); }
    vec3 operator+(const vec3 &v) const { return vec3(x + v.x, y + v.y, z + v.z); }
    void operator+=(const vec3 &v)
    {
        x += v.x;
        y += v.y;
        z += v.z;
    }
    vec3 operator-(const vec3 &v) const { return vec3(x - v.x, y - v.y, z - v.z); }
    vec3 operator*(const vec3 &v) const { return vec3(x * v.x, y * v.y, z * v.z); }
    vec3 operator-() const { return vec3(-x, -y, -z); }
    vec3 normalize() const { return (*this) * (1 / (Length() + 0.000001)); }
    float Length() const { return sqrtf(x * x + y * y + z * z); }
    operator float *() { return &x; }
};

float dot(const vec3 &v1, const vec3 &v2)
{
    return (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z);
}

vec3 cross(const vec3 &v1, const vec3 &v2)
{
    return vec3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
}

struct Material
{
    bool rough = false;
    bool reflective = false;
    bool refractive = false;

    vec3 ka, kd, ks;
    float shininess;

    vec3 n, k, F0;

    float n_;

    Material(vec3 _ka, vec3 _kd, vec3 _ks, float _shininess) : ka(_ka), kd(_kd), ks(_ks)
    {
        shininess = _shininess;
        rough = true;
    }

    Material(vec3 _n, vec3 _k)
    {
        n = _n;
        k = _k;
        F0.x = ((n.x - 1) * (n.x - 1) + k.x * k.x) / ((n.x + 1) * (n.x + 1) + k.x * k.x);
        F0.y = ((n.y - 1) * (n.y - 1) + k.y * k.y) / ((n.y + 1) * (n.y + 1) + k.y * k.y);
        F0.z = ((n.z - 1) * (n.z - 1) + k.z * k.z) / ((n.z + 1) * (n.z + 1) + k.z * k.z);
        reflective = true;
    };

    Material(float a)
    {
        n_ = a;
        refractive = true;
    }

    vec3 shade(vec3 normal, vec3 viewDir, vec3 lightDir, vec3 inRad)
    {
        vec3 reflRad(0, 0, 0);
        float cosTheta = dot(normal, lightDir);
        if (cosTheta < 0)
            return reflRad;
        reflRad = inRad * kd * cosTheta;
        vec3 halfway = (viewDir + lightDir).normalize();
        float cosDelta = dot(normal, halfway);
        if (cosDelta < 0)
            return reflRad;
        return reflRad + inRad * ks * pow(cosDelta, shininess);
    };

    vec3 reflect(vec3 inDir, vec3 normal)
    {
        return inDir - normal * dot(normal, inDir) * 2.0f;
    };

    vec3 fresnel(vec3 inDir, vec3 normal)
    {
        float cosa = -dot(inDir, normal);
        vec3 one(1, 1, 1);
        return F0 + (one - F0) * pow(1 - cosa, 5);
    };

    vec3 refract(vec3 inDir, vec3 normal, float n0)
    {
        float cosa = -dot(inDir, normal);
        float disc = 1 - (1 - cosa * cosa) / n0 / n0;
        if (disc < 0)
            return vec3(0, 0, 0);
        return inDir / n0 + normal * (cosa / n0 - sqrt(disc));
    }
};

struct Hit
{
    float t;
    vec3 position;
    vec3 normal;
    Material *material;
    Hit() { t = -1; }
};

struct Ray
{
    vec3 start, dir;
    bool out;
    Ray(vec3 _start, vec3 _dir, bool a = false)
    {
        start = _start;
        dir = _dir.normalize();
        out = a;
    }
};

class Intersectable
{
protected:
    Material *material;

public:
    virtual Hit intersect(const Ray &ray) = 0;
};

struct Sphere : public Intersectable
{
    vec3 center;
    float radius;

    Sphere(const vec3 &_center, float _radius, Material *_material)
    {
        center = _center;
        radius = _radius;
        material = _material;
    }
    Hit intersect(const Ray &ray)
    {
        Hit hit;
        vec3 dist = ray.start - center;
        float b = dot(dist, ray.dir) * 2.0f;
        float a = dot(ray.dir, ray.dir);
        float c = dot(dist, dist) - radius * radius;
        float discr = b * b - 4.0f * a * c;
        if (discr < 0)
            return hit;
        float sqrt_discr = sqrtf(discr);
        float t1 = (-b + sqrt_discr) / 2.0f / a;
        float t2 = (-b - sqrt_discr) / 2.0f / a;
        if (t1 <= 0 && t2 <= 0)
            return hit;
        if (t1 <= 0 && t2 > 0)
            hit.t = t2;
        else if (t2 <= 0 && t1 > 0)
            hit.t = t1;
        else if (t1 < t2)
            hit.t = t1;
        else
            hit.t = t2;
        hit.position = ray.start + ray.dir * hit.t;
        hit.normal = (hit.position - center) / radius;
        if (dot(hit.normal, ray.dir) > 0)
            hit.normal = hit.normal * (-1);
        hit.material = material;
        return hit;
    }
};

struct Paraboloid : public Intersectable
{
    vec3 start;
    float radius;
    float height;

    Paraboloid(const vec3 &c, float rad, float hi, Material *mat)
    {
        start = c;
        radius = rad;
        height = hi;
        material = mat;
    }

    Hit intersect(const Ray &ray)
    {
        Hit hit;

        float b = 2 * (ray.start.x * ray.dir.x - start.x * ray.dir.x + ray.start.y * ray.dir.y - start.y * ray.dir.y) / (radius * radius) - ray.dir.z;
        float a = (ray.dir.x * ray.dir.x + ray.dir.y * ray.dir.y) / (radius * radius);
        float c = (start.x * start.x + ray.start.x * ray.start.x - 2 * start.x * ray.start.x + start.y * start.y + ray.start.y * ray.start.y - 2 * start.y * ray.start.y) / (radius * radius) + start.z - ray.start.z;
        float discr = b * b - 4.0f * a * c;
        if (discr < 0)
            return hit;
        Hit ignore = hit;
        float sqrt_discr = sqrtf(discr);
        float t1 = (-b + sqrt_discr) / 2.0f / a;
        float t2 = (-b - sqrt_discr) / 2.0f / a;
        if (t1 <= 0 && t2 <= 0)
            return hit;
        if (t1 <= 0 && t2 > 0)
            hit.t = t2;
        else if (t2 <= 0 && t1 > 0)
            hit.t = t1;
        else if (t1 < t2)
            hit.t = t1;
        else
            hit.t = t2;
        hit.position = ray.start + ray.dir * hit.t;
        if (hit.position.z > height + start.z)
            return ignore;
        hit.normal.x = (hit.position.x - start.x) / (radius * radius);
        hit.normal.y = (hit.position.y - start.y) / (radius * radius);
        hit.normal.z = -0.5f;
        if (dot(hit.normal, ray.dir) > 0)
            hit.normal = hit.normal * (-1);
        hit.material = material;
        return hit;
    }
};

struct ParaboloidAquarium : public Intersectable
{
    vec3 start;
    float radius;
    float height;
    Material *water;
    std::vector<Intersectable *> paraboloids;

    ParaboloidAquarium(const vec3 &c, float rad, float hi)
    {
        start = c;
        radius = rad;
        height = hi;
        material = new Material(1.5f);
        water = new Material(1.3f);
        CreateAquarium();
    }

    void CreateAquarium()
    {
        paraboloids.push_back(new Paraboloid(start, radius, height, material));
        paraboloids.push_back(new Paraboloid(vec3(start.x, start.y, start.z + 0.001f), radius - 0.001f, height - 0.001f, water));
    }

    Hit intersect(const Ray &ray)
    {
        Hit hit;
        for (Intersectable *i : paraboloids)
        {
            Hit h = i->intersect(ray);
            if (h.t > 0 && (hit.t < 0 || h.t < hit.t))
                hit = h;
        };
        return hit;
    }
};

struct Cone : public Intersectable
{
    vec3 start;
    float radius;
    float height;

    Cone(const vec3 &c, float rad, float hi, Material *mat)
    {
        start = c;
        radius = rad;
        height = hi;
        material = mat;
    }

    Hit intersect(const Ray &ray)
    {
        Hit hit;

        float b = 2 * (ray.start.x * ray.dir.x - start.x * ray.dir.x + ray.start.y * ray.dir.y - start.y * ray.dir.y) / (radius * radius) - 2 * (ray.start.z * ray.dir.z - height * ray.dir.z) / (height * height);
        float a = (ray.dir.x * ray.dir.x + ray.dir.y * ray.dir.y) / (radius * radius) - ray.dir.z * ray.dir.z / (height * height);
        float c = (start.x * start.x + ray.start.x * ray.start.x - 2 * start.x * ray.start.x + start.y * start.y + ray.start.y * ray.start.y - 2 * start.y * ray.start.y) / (radius * radius) - (height * height + ray.start.z * ray.start.z - 2 * height * ray.start.z) / (height * height);
        float discr = b * b - 4.0f * a * c;
        if (discr < 0)
            return hit;
        Hit ignore = hit;
        float sqrt_discr = sqrtf(discr);
        float t1 = (-b + sqrt_discr) / 2.0f / a;
        float t2 = (-b - sqrt_discr) / 2.0f / a;
        if (t1 <= 0 && t2 <= 0)
            return hit;
        if (t1 <= 0 && t2 > 0)
            hit.t = t2;
        else if (t2 <= 0 && t1 > 0)
            hit.t = t1;
        else if (t1 < t2)
            hit.t = t1;
        else
            hit.t = t2;
        hit.position = ray.start + ray.dir * hit.t;
        if (hit.position.z > height || hit.position.z < 0)
            return ignore;
        hit.normal.x = (hit.position.x - start.x) / (radius * radius);
        hit.normal.y = (hit.position.y - start.y) / (radius * radius);
        hit.normal.z = (height - hit.position.z) / (height * height);
        hit.material = material;
        return hit;
    }
};

struct Plane : public Intersectable
{
    vec3 start;
    vec3 normal;

    Plane(const vec3 &c, const vec3 &d, Material *mat)
    {
        start = c;
        normal = d;
        material = mat;
    }

    Hit intersect(const Ray &ray)
    {
        Hit hit;
        float b = normal.x * ray.start.x - normal.x * start.x + normal.y * ray.start.y - normal.y * start.y + normal.z * ray.start.z - normal.z * start.z;
        float a = normal.x * ray.dir.x + normal.y * ray.dir.y + normal.z * ray.dir.z;
        hit.t = -1 * b / a;
        hit.position = ray.start + ray.dir * hit.t;
        hit.normal = normal;
        hit.material = material;
        return hit;
    }
};

struct FiniteVerticalPlane : public Intersectable
{
    vec3 start;
    vec3 normal;
    float width;
    float height;

    FiniteVerticalPlane(const vec3 &c, const vec3 &d, float a, float b, Material *mat)
    {
        start = c;
        normal = d;
        width = a;
        height = b;
        material = mat;
    }

    Hit intersect(const Ray &ray)
    {
        Hit hit;
        Hit ignore;
        float b = normal.x * ray.start.x - normal.x * start.x + normal.y * ray.start.y - normal.y * start.y + normal.z * ray.start.z - normal.z * start.z;
        float a = normal.x * ray.dir.x + normal.y * ray.dir.y + normal.z * ray.dir.z;
        hit.t = -1 * b / a;
        hit.position = ray.start + ray.dir * hit.t;
        hit.normal = normal;
        if (dot(hit.normal, ray.dir) > 0)
            hit.normal = hit.normal * (-1);
        if (sqrtf((hit.position - start).x * (hit.position - start).x + (hit.position - start).y * (hit.position - start).y) > width || sqrtf((hit.position - start).z * (hit.position - start).z) > height)
            return ignore;
        hit.material = material;
        return hit;
    }
};

struct TexturedFiniteVerticalPlane : public Intersectable
{
    vec3 start;
    vec3 normal;
    float width;
    float height;
    Material *mat2;

    TexturedFiniteVerticalPlane(const vec3 &c, const vec3 &d, float a, float b, Material *m2, Material *mat = new Material(vec3(0.0f, 0.0f, 0.0f), vec3(0.0f, 0.0f, 0.0f), vec3(0.1f, 0.1f, 0.1f), 0.1f))
    {
        start = c;
        normal = d;
        width = a;
        height = b;
        material = mat;
        mat2 = m2;
    }

    Hit intersect(const Ray &ray)
    {
        Hit hit;
        Hit ignore;
        float b = normal.x * ray.start.x - normal.x * start.x + normal.y * ray.start.y - normal.y * start.y + normal.z * ray.start.z - normal.z * start.z;
        float a = normal.x * ray.dir.x + normal.y * ray.dir.y + normal.z * ray.dir.z;
        hit.t = -1 * b / a;
        hit.position = ray.start + ray.dir * hit.t;
        hit.normal = normal;
        if (sqrtf((hit.position - start).x * (hit.position - start).x + (hit.position - start).y * (hit.position - start).y) > width || sqrtf((hit.position - start).z * (hit.position - start).z) > height)
            return ignore;
        hit.material = material;
        if ((fabs(fmod(hit.position.x / 0.25, 2.0)) >= 1.0 || fabs(fmod(hit.position.y / 0.25, 2.0)) >= 1.0) && (fabs(fmod(hit.position.z / 0.25, 2.0)) <= 1.0))
            hit.material = mat2;
        return hit;
    }
};

struct FiniteHorizontalPlane : public Intersectable
{
    vec3 start;
    vec3 normal;
    float width;
    float height;

    FiniteHorizontalPlane(const vec3 &c, const vec3 &d, float a, float b, Material *mat)
    {
        start = c;
        normal = d;
        width = a;
        height = b;
        material = mat;
    }

    Hit intersect(const Ray &ray)
    {
        Hit hit;
        Hit ignore;
        float b = normal.x * ray.start.x - normal.x * start.x + normal.y * ray.start.y - normal.y * start.y + normal.z * ray.start.z - normal.z * start.z;
        float a = normal.x * ray.dir.x + normal.y * ray.dir.y + normal.z * ray.dir.z;
        hit.t = -1 * b / a;
        hit.position = ray.start + ray.dir * hit.t;
        hit.normal = normal;
        if (dot(hit.normal, ray.dir) > 0)
            hit.normal = hit.normal * (-1);
        if (sqrtf((hit.position - start).x * (hit.position - start).x) > width || sqrtf((hit.position - start).y * (hit.position - start).y) > height)
            return ignore;
        hit.material = material;
        return hit;
    }
};

struct TexturedFiniteHorizontalPlane : public Intersectable
{
    vec3 start;
    vec3 normal;
    float width;
    float height;
    Material *mat2;

    TexturedFiniteHorizontalPlane(const vec3 &c, const vec3 &d, float a, float b, Material *m2, Material *mat = new Material(vec3(0.0f, 0.0f, 0.0f), vec3(0.0f, 0.0f, 0.0f), vec3(0.1f, 0.1f, 0.1f), 0.1f))
    {
        start = c;
        normal = d;
        width = a;
        height = b;
        material = mat;
        mat2 = m2;
    }

    Hit intersect(const Ray &ray)
    {
        Hit hit;
        Hit ignore;
        float b = normal.x * ray.start.x - normal.x * start.x + normal.y * ray.start.y - normal.y * start.y + normal.z * ray.start.z - normal.z * start.z;
        float a = normal.x * ray.dir.x + normal.y * ray.dir.y + normal.z * ray.dir.z;
        hit.t = -1 * b / a;
        hit.position = ray.start + ray.dir * hit.t;
        hit.normal = normal;
        if (sqrtf((hit.position - start).x * (hit.position - start).x) > width || sqrtf((hit.position - start).y * (hit.position - start).y) > height)
            return ignore;
        hit.material = material;
        if ((fabs(fmod(hit.position.x / 0.25f, 2.0f)) >= 1.0f) && (fabs(fmod(hit.position.y / 0.25f, 2.0f)) <= 1.0f))
            hit.material = mat2;
        return hit;
    }
};

struct Cuboid : public Intersectable
{
    vec3 start;
    float width;
    float height;
    float depth;
    std::vector<Intersectable *> planes;

    Cuboid(const vec3 &c, float wid, float hi, float dep, Material *mat)
    {
        start = c;
        width = wid;
        height = hi;
        depth = dep;
        material = mat;
        BuildPlanes();
    }

    void BuildPlanes()
    {
        planes.push_back(new FiniteHorizontalPlane(vec3(start.x, start.y, start.z + 0.001f), vec3(0.0f, 0.0f, 1.0f), width / 2, depth / 2, material));
        planes.push_back(new FiniteVerticalPlane(vec3(start.x, start.y - depth / 2, start.z + height / 2), vec3(0, -1, 0), width / 2, height / 2, material));
        planes.push_back(new FiniteVerticalPlane(vec3(start.x, start.y + depth / 2, start.z + height / 2), vec3(0, 1, 0), width / 2, height / 2, material));
        planes.push_back(new FiniteVerticalPlane(vec3(start.x - width / 2, start.y, start.z + height / 2), vec3(-1, 0, 0), depth / 2, height / 2, material));
        planes.push_back(new FiniteVerticalPlane(vec3(start.x + width / 2, start.y, start.z + height / 2), vec3(1, 0, 0), depth / 2, height / 2, material));
    }

    Hit intersect(const Ray &ray)
    {
        Hit hit;
        for (Intersectable *i : planes)
        {
            Hit h = i->intersect(ray);
            if (h.t > 0 && (hit.t < 0 || h.t < hit.t))
                hit = h;
        };
        return hit;
    }
};

struct CuboidAquarium : public Intersectable
{
    vec3 start;
    float width;
    float height;
    float depth;
    Material *water;
    std::vector<Intersectable *> cuboids;

    CuboidAquarium(const vec3 &c, float wid, float hi, float dep)
    {
        start = c;
        width = wid;
        height = hi;
        depth = dep;
        material = new Material(1.5f);
        water = new Material(1.3f);
        CreateAquarium();
    }

    void CreateAquarium()
    {
        cuboids.push_back(new Cuboid(start, width, height, depth, material));
        cuboids.push_back(new Cuboid(vec3(start.x, start.y, start.z + 0.001f), width - 0.001f, height - 0.001f, depth - 0.001f, water));
    }

    Hit intersect(const Ray &ray)
    {
        Hit hit;
        for (Intersectable *i : cuboids)
        {
            Hit h = i->intersect(ray);
            if (h.t > 0 && (hit.t < 0 || h.t < hit.t))
                hit = h;
        };
        return hit;
    }
};

struct Room : public Intersectable
{
    std::vector<Intersectable *> walls;
    float size;
    Material *mat2;

    Room(float s, Material *m2, Material *mat = new Material(vec3(0.0f, 0.0f, 0.0f), vec3(0.0f, 0.0f, 0.0f), vec3(0.1f, 0.1f, 0.1f), 0.1f))
    {
        size = s;
        material = mat;
        mat2 = m2;
        BuildWalls();
    }

    void BuildWalls()
    {
        walls.push_back(new TexturedFiniteHorizontalPlane(vec3(0.0f, 0.0f, 0.0f), vec3(0.0f, 0.0f, 1.0f), size / 2, size / 2, mat2, material));
        walls.push_back(new TexturedFiniteHorizontalPlane(vec3(0.0f, 0.0f, size), vec3(0.0f, 0.0f, -1.0f), size / 2, size / 2, mat2, material));
        walls.push_back(new TexturedFiniteVerticalPlane(vec3(-size / 2, 0.0f, size / 2), vec3(1.0f, 0.0f, 0.0f), size / 2, size / 2, mat2, material));
        walls.push_back(new TexturedFiniteVerticalPlane(vec3(size / 2, 0.0f, size / 2), vec3(-1.0f, 0.0f, 0.0f), size / 2, size / 2, mat2, material));
        walls.push_back(new TexturedFiniteVerticalPlane(vec3(0, size / 2, size / 2), vec3(0.0f, -1.0f, 0.0f), size / 2, size / 2, mat2, material));
        walls.push_back(new TexturedFiniteVerticalPlane(vec3(0, -size / 2, size / 2), vec3(0.0f, 1.0f, 0.0f), size / 2, size / 2, mat2, material));
    }

    Hit intersect(const Ray &ray)
    {
        Hit hit;
        for (Intersectable *i : walls)
        {
            Hit h = i->intersect(ray);
            if (h.t > 0 && (hit.t < 0 || h.t < hit.t))
                hit = h;
        };
        return hit;
    }
};

class Camera
{
    vec3 eye, lookat, right, up;

public:
    void set(vec3 _eye, vec3 _lookat, vec3 vup, double fov)
    {
        eye = _eye;
        lookat = _lookat;
        vec3 w = eye - lookat;
        float f = w.Length();
        right = cross(vup, w).normalize() * f * tan(fov / 2);
        up = cross(w, right).normalize() * f * tan(fov / 2);
    }

    Ray getray(int X, int Y, int screenw = windowWidth, int screenh = windowHeight)
    {
        vec3 dir = lookat + right * (2.0 * (X + 0.5) / screenw - 1) + up * (2.0 * (Y + 0.5) / screenh - 1) - eye;
        return Ray(eye, dir);
    }

    Ray getray_primitivefish(int X, int Y, int screenw = windowWidth, int screenh = windowHeight)
    {
        float theta = ((float)Y / (float)screenh) * M_PI;
        float phi = ((float)X / (float)screenw) * 2.0f * M_PI;
        vec3 dir(sin(theta) * cos(phi), sin(theta) * sin(phi), -1.0f * cos(theta));
        return Ray(eye, dir);
    }

    Ray getray_cylindrical(int X, int Y, int screenw = windowWidth, int screenh = windowHeight)
    {
        int x_ = X - (screenw / 2);
        int y_ = (screenh / 2) - Y;
        float w = fabsf(x_ / (screenw / 2.0));
        float phi = M_PI / 2.0 - w * M_PI;
        if (x_ < 0)
            phi = M_PI / 2.0 + w * M_PI;
        vec3 dir(cos(phi), sin(phi), -1.0f * y_ / (screenh / 2.0f));
        return Ray(eye, dir);
    }

    Ray getray_littleplanet(int X, int Y, int screenw = windowWidth, int screenh = windowHeight)
    {
        int x_ = X - (screenw / 2);
        int y_ = (screenh / 2) - Y;
        float R = sqrtf(x_ * x_ + y_ * y_);
        float maxR = sqrtf((screenw / 2) * (screenw / 2) + (screenh / 2) * (screenh / 2));
        float theta = (R / maxR) * M_PI;
        float phi = -1.0f * atan2f(y_, x_);
        vec3 dir(sin(theta) * cos(phi), cos(theta), sin(theta) * sin(phi));
        return Ray(eye, dir);
    }
};

struct Light
{
    vec3 direction;
    vec3 Le;
    Light(vec3 _direction, vec3 _Le)
    {
        direction = _direction.normalize();
        Le = _Le;
    }
};

class Scene
{
    std::vector<Intersectable *> objects;
    std::vector<Light *> lights;
    Camera camera;
    vec3 La;
    int maxdepth = 6;

public:
    void build()
    {
        vec3 eye = vec3(0.0, -2.0, 0.5);
        vec3 vup = vec3(0, 0, 1);
        vec3 lookat = vec3(0, 0, 0.5);
        float fov = 60 * M_PI / 180;
        camera.set(eye, lookat, vup, fov);
        La = vec3(0.1f, 0.1f, 0.1f);

        objects.push_back(new Room(8.0f, new Material(vec3(1.0f, 1.0f, 1.0f) * M_PI, vec3(1.0f, 1.0f, 1.0f), vec3(0.1f, 0.1f, 0.1f), 1.0f)));

        lights.push_back(new Light(vec3(-3.0f, -1.0f, 8.0f), vec3(0.8f, 0.0f, 1.0f)));
        lights.push_back(new Light(vec3(3.0f, -1.0f, 8.0f), vec3(0.94f, 0.9f, 0.55f)));

        objects.push_back(new Cone(vec3(0.0f, 0.0f, 0.0f), 0.7f, 0.8f, new Material(vec3(0.17f, 0.35f, 1.5f), vec3(3.1f, 2.7f, 1.9f))));
        objects.push_back(new ParaboloidAquarium(vec3(0.0, -2.0, 0.0), 0.9f, 1.0f));
    }

    void render(vec3 image[])
    {
#pragma omp parallel for
        for (int Y = 0; Y < windowHeight; Y++)
        {
            if (Y == windowHeight / 2)
            {
                objects.pop_back();
                objects.push_back(new CuboidAquarium(vec3(0.0f, -2.0f, 0.0f), 0.9f, 0.9f, 0.6f));
            }
            for (int X = 0; X < windowWidth; X++)
            {
                if (X < windowWidth / 2)
                {
                    if (Y < windowHeight / 2)
                    {
                        image[Y * windowWidth + X] = trace(camera.getray(X, Y, windowWidth / 2, windowHeight / 2));
                    }
                    else
                    {
                        image[Y * windowWidth + X] = trace(camera.getray(X, Y - windowHeight / 2, windowWidth / 2, windowHeight / 2));
                    }
                }
                else
                {
                    if (Y < windowHeight / 2)
                    {
                        image[Y * windowWidth + X] = trace(camera.getray_littleplanet(X - windowWidth / 2, Y, windowWidth / 2, windowHeight / 2));
                    }
                    else
                    {
                        image[Y * windowWidth + X] = trace(camera.getray_littleplanet(X - windowWidth / 2, Y - windowHeight / 2, windowWidth / 2, windowHeight / 2));
                    }
                }
            }
        }
    }

    Hit firstIntersect(Ray ray)
    {
        Hit bestHit;
        for (Intersectable *object : objects)
        {
            Hit hit = object->intersect(ray);
            if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))
                bestHit = hit;
        }
        return bestHit;
    }

    vec3 trace(Ray ray, int d = 0)
    {
        if (d > maxdepth)
            return La;
        Hit hit = firstIntersect(ray);
        if (hit.t < 0)
            return La;
        vec3 outRad(0, 0, 0);
        if (hit.material->rough)
        {
            vec3 outRadiance = hit.material->ka * La;
            for (Light *light : lights)
            {
                Ray shadowRay(hit.position + hit.normal * 0.01, light->direction - hit.position);
                Hit shadowHit = firstIntersect(shadowRay);
                if (shadowHit.t < 0 || shadowHit.t > (hit.position - light->direction).Length())
                    outRadiance += hit.material->shade(hit.normal, -ray.dir, light->direction, light->Le);
            }
            outRad = outRadiance;
        }
        if (hit.material->reflective)
        {
            vec3 reflectionDir = hit.material->reflect(ray.dir, hit.normal);
            Ray reflectRay(hit.position + hit.normal * 0.01, reflectionDir, ray.out);
            outRad += trace(reflectRay, d + 1) * hit.material->fresnel(ray.dir, hit.normal);
        }
        if (hit.material->refractive)
        {
            float ior = (ray.out) ? hit.material->n_ : (1 / hit.material->n_);
            vec3 refractionDir = hit.material->refract(ray.dir, hit.normal, ior);
            if (refractionDir.Length() > 0)
            {
                Ray refractRay(hit.position - hit.normal * 0.01, refractionDir, !ray.out);
                outRad += trace(refractRay, d + 1) * (vec3(1, 1, 1) - hit.material->fresnel(ray.dir, hit.normal));
            }
        }
        return outRad;
    }
};

Scene scene;

void getErrorInfo(unsigned int handle)
{
    int logLen, written;
    glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
    if (logLen > 0)
    {
        char *log = new char[logLen];
        glGetShaderInfoLog(handle, logLen, &written, log);
        printf("Shader log:\n%s", log);
        delete log;
    }
}

void checkShader(unsigned int shader, const char *message)
{
    int OK;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
    if (!OK)
    {
        printf("%s!\n", message);
        getErrorInfo(shader);
    }
}

void checkLinking(unsigned int program)
{
    int OK;
    glGetProgramiv(program, GL_LINK_STATUS, &OK);
    if (!OK)
    {
        printf("Failed to link shader program!\n");
        getErrorInfo(program);
    }
}

const char *vertexSource = R"(
    #version 330
    precision highp float;
 
    layout(location = 0) in vec2 vertexPosition;    
    out vec2 texcoord;
 
    void main() {
        texcoord = (vertexPosition + vec2(1, 1))/2;                            
        gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1);         
    }
)";

const char *fragmentSource = R"(
    #version 330
    precision highp float;
 
    uniform sampler2D textureUnit;
    in  vec2 texcoord;            
    out vec4 fragmentColor;        
 
    void main() {
        fragmentColor = texture(textureUnit, texcoord); 
    }
)";

unsigned int shaderProgram;

class FullScreenTexturedQuad
{
    unsigned int vao, textureId;

public:
    void Create(vec3 image[])
    {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        unsigned int vbo;
        glGenBuffers(1, &vbo);

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        float vertexCoords[] = {-1, -1, 1, -1, 1, 1, -1, 1};
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

        glGenTextures(1, &textureId);
        glBindTexture(GL_TEXTURE_2D, textureId);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowWidth, windowHeight, 0, GL_RGB, GL_FLOAT, image);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }

    void Draw()
    {
        glBindVertexArray(vao);
        int location = glGetUniformLocation(shaderProgram, "textureUnit");
        if (location >= 0)
        {
            glUniform1i(location, 0);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, textureId);
        }
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    }
};

FullScreenTexturedQuad fullScreenTexturedQuad;
vec3 image[windowWidth * windowHeight];

void onInitialization()
{
    glViewport(0, 0, windowWidth, windowHeight);
    scene.build();
    scene.render(image);
    fullScreenTexturedQuad.Create(image);

    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    if (!vertexShader)
    {
        printf("Error in vertex shader creation\n");
        exit(1);
    }
    glShaderSource(vertexShader, 1, &vertexSource, NULL);
    glCompileShader(vertexShader);
    checkShader(vertexShader, "Vertex shader error");

    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    if (!fragmentShader)
    {
        printf("Error in fragment shader creation\n");
        exit(1);
    }
    glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
    glCompileShader(fragmentShader);
    checkShader(fragmentShader, "Fragment shader error");

    shaderProgram = glCreateProgram();
    if (!shaderProgram)
    {
        printf("Error in shader program creation\n");
        exit(1);
    }
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);

    glBindFragDataLocation(shaderProgram, 0, "fragmentColor");

    glLinkProgram(shaderProgram);
    checkLinking(shaderProgram);
    glUseProgram(shaderProgram);
}

void onExit()
{
    glDeleteProgram(shaderProgram);
    printf("exit");
}

void onDisplay()
{
    fullScreenTexturedQuad.Draw();
    glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY)
{
}

void onKeyboardUp(unsigned char key, int pX, int pY)
{
}

void onMouse(int button, int state, int pX, int pY)
{
}

void onMouseMotion(int pX, int pY)
{
}

void onIdle()
{
}

int main(int argc, char *argv[])
{
    glutInit(&argc, argv);
#if !defined(__APPLE__)
    glutInitContextVersion(majorVersion, minorVersion);
#endif
    glutInitWindowSize(windowWidth, windowHeight);
    glutInitWindowPosition(100, 100);
#if defined(__APPLE__)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_3_CORE_PROFILE);
#else
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
    glutCreateWindow(argv[0]);

#if !defined(__APPLE__)
    glewExperimental = true;
    glewInit();
#endif

    printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
    printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
    printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
    glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
    glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
    printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
    printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

    onInitialization();

    glutDisplayFunc(onDisplay);
    glutMouseFunc(onMouse);
    glutIdleFunc(onIdle);
    glutKeyboardFunc(onKeyboard);
    glutKeyboardUpFunc(onKeyboardUp);
    glutMotionFunc(onMouseMotion);

    glutMainLoop();
    onExit();
    return 1;
}
