#include <metal_stdlib>
using namespace metal;

// DEVICE VALIDATION REQUIRED: runtime compilation, float parity, timing and visuals.
// Scalar formulas mirror Math.swift. No fast-math contract is assumed.
struct D { float v, d; D(float v=0, float d=0):v(v),d(d) {} };
D operator+(D a,D b) { return D(a.v+b.v,a.d+b.d); }
D operator-(D a,D b) { return D(a.v-b.v,a.d-b.d); }
D operator-(D a) { return D(-a.v,-a.d); }
D operator*(D a,D b) { return D(a.v*b.v,a.d*b.v+a.v*b.d); }
D operator/(D a,D b) { return D(a.v/b.v,(a.d*b.v-a.v*b.d)/(b.v*b.v)); }
D dexp(D a) { float e=exp(a.v); return D(e,e*a.d); }
D dsqrt(D a) { float s=sqrt(a.v); return D(s,a.d/(2*s)); }
struct Gaussian { float p[14]; };
// camera: world->optical row-major 9, world centre 3, fx fy cx cy, width height.
struct Camera { float p[18]; };
struct Projection { float u,v,z,xx,xy,yy,radius,opacity; };
struct Settings { uint count,width,height,offset; };
constant uint tileSize=16;
constant uint tileCapacity=64;

bool project(thread D* p, constant Camera& camera, thread D* out) {
    D n(0); for(uint k=6;k<10;k++) n=n+p[k]*p[k]; n=dsqrt(n);
    D w=p[6]/n,x=p[7]/n,y=p[8]/n,z=p[9]/n;
    D r[9]={D(1)-D(2)*(y*y+z*z),D(2)*(x*y-w*z),D(2)*(x*z+w*y),
            D(2)*(x*y+w*z),D(1)-D(2)*(x*x+z*z),D(2)*(y*z-w*x),
            D(2)*(x*z-w*y),D(2)*(y*z+w*x),D(1)-D(2)*(x*x+y*y)};
    D cov[9];
    for(uint i=0;i<9;i++) {
        cov[i]=D(0);
        for(uint k=0;k<3;k++) cov[i]=cov[i]+r[(i/3)*3+k]*dexp(D(2)*p[3+k])*r[(i%3)*3+k];
    }
    D t[3];
    for(uint i=0;i<3;i++) {
        t[i]=D(0);
        for(uint k=0;k<3;k++) t[i]=t[i]+D(camera.p[i*3+k])*(p[k]-D(camera.p[9+k]));
    }
    if(t[2].v<=0.01f) return false;
    D fx(camera.p[12]),fy(camera.p[13]),zz=t[2];
    D j[6]={fx/zz,D(0),-fx*t[0]/(zz*zz),D(0),fy/zz,-fy*t[1]/(zz*zz)};
    D a[6];
    for(uint i=0;i<6;i++) {
        a[i]=D(0);
        for(uint k=0;k<3;k++) a[i]=a[i]+j[(i/3)*3+k]*D(camera.p[k*3+i%3]);
    }
    out[0]=fx*t[0]/zz+D(camera.p[14]); out[1]=fy*t[1]/zz+D(camera.p[15]); out[2]=zz;
    for(uint i=0;i<3;i++) {
        uint row=i==2?1:0, col=i==0?0:1;
        D c(row==col?0.3f:0.f);
        for(uint k=0;k<3;k++) for(uint l=0;l<3;l++) c=c+a[row*3+k]*cov[k*3+l]*a[col*3+l];
        out[3+i]=c;
    }
    for(uint i=0;i<6;i++) if(!isfinite(out[i].v)) return false;
    float determinant=out[3].v*out[5].v-out[4].v*out[4].v;
    return isfinite(determinant) && determinant>0;
}

kernel void projectGaussians(device const Gaussian* splats [[buffer(0)]],
                             device Projection* projected [[buffer(1)]],
                             constant Camera& camera [[buffer(2)]],
                             constant Settings& settings [[buffer(3)]], uint id [[thread_position_in_grid]]) {
    if(id>=settings.count) return;
    D p[14],c[6]; for(uint i=0;i<14;i++) p[i]=D(splats[id].p[i]);
    Projection o={0,0,-1,0,0,0,0,0};
    if(project(p,camera,c)) {
        o.u=c[0].v; o.v=c[1].v; o.z=c[2].v; o.xx=c[3].v; o.xy=c[4].v; o.yy=c[5].v;
        o.radius=3*sqrt(0.5f*(o.xx+o.yy+sqrt((o.xx-o.yy)*(o.xx-o.yy)+4*o.xy*o.xy)));
        o.opacity=1/(1+exp(-p[10].v));
    }
    projected[id]=o;
}

// One owner per tile: deterministic insertion sort by depth, then index for ties.
// Bounded memory, nearest 64 retained; overflow is surfaced as a coverage warning.
kernel void binAndSort(device const Projection* projected [[buffer(0)]],
                       device uint* bins [[buffer(1)]], device uint* overflow [[buffer(2)]],
                       constant Settings& settings [[buffer(3)]], uint tile [[thread_position_in_grid]]) {
    uint columns=(settings.width+tileSize-1)/tileSize, rows=(settings.height+tileSize-1)/tileSize;
    if(tile>=columns*rows) return;
    float2 lo=float2(tile%columns,tile/columns)*float(tileSize), hi=lo+float(tileSize);
    uint ids[tileCapacity],count=0,dropped=0;
    for(uint i=0;i<settings.count;i++) {
        Projection p=projected[i];
        if(p.z<=0 || p.u+p.radius<lo.x || p.u-p.radius>hi.x || p.v+p.radius<lo.y || p.v-p.radius>hi.y) continue;
        uint at=count;
        while(at>0 && projected[ids[at-1]].z>p.z) at--;
        if(count==tileCapacity) dropped++;
        if(at>=tileCapacity) continue;
        uint end=min(count,tileCapacity-1);
        for(uint k=end;k>at;k--) ids[k]=ids[k-1];
        ids[at]=i; count=min(count+1,tileCapacity);
    }
    uint base=tile*(tileCapacity+1); bins[base]=count;
    for(uint k=0;k<count;k++) bins[base+1+k]=ids[k];
    overflow[tile]=dropped;
}

kernel void compositeGaussians(device const Gaussian* splats [[buffer(0)]],
                               device const Projection* projected [[buffer(1)]],
                               device const uint* bins [[buffer(2)]],
                               constant Settings& settings [[buffer(3)]],
                               device const uint* coverage [[buffer(4)]],
                               constant float* viewToSensor [[buffer(5)]],
                               device const uint* overflow [[buffer(6)]],
                               texture2d<float,access::write> output [[texture(0)]],
                               uint2 pixel [[thread_position_in_grid]]) {
    if(pixel.x>=output.get_width() || pixel.y>=output.get_height()) return;
    float2 uv=(float2(pixel)+0.5f)/float2(output.get_width(),output.get_height());
    float2 sensor=float2(viewToSensor[0]*uv.x+viewToSensor[2]*uv.y+viewToSensor[4],
                         viewToSensor[1]*uv.x+viewToSensor[3]*uv.y+viewToSensor[5]);
    if(any(sensor<0) || any(sensor>=1)) { output.write(float4(0),pixel); return; }
    float2 sample=sensor*float2(settings.width,settings.height);
    uint columns=(settings.width+tileSize-1)/tileSize;
    uint tile=uint(sample.y)/tileSize*columns+uint(sample.x)/tileSize;
    uint base=tile*(tileCapacity+1);
    float3 rgb=0; float transmittance=1;
    for(uint k=0;k<bins[base];k++) {
        uint i=bins[base+1+k]; Projection p=projected[i];
        float2 d=sample-float2(p.u,p.v);
        float q=(p.yy*d.x*d.x-2*p.xy*d.x*d.y+p.xx*d.y*d.y)/(p.xx*p.yy-p.xy*p.xy);
        if(q>9) continue;
        float alpha=min(0.99f,p.opacity*exp(-0.5f*q));
        rgb+=transmittance*alpha*float3(splats[i].p[11],splats[i].p[12],splats[i].p[13]);
        transmittance*=1-alpha;
    }
    uint cell=min(uint(sensor.y*24),23u)*32+min(uint(sensor.x*32),31u);
    if(coverage[cell]==0 || overflow[tile]>0) {
        // Premultiplied cyan hint; AR camera remains visible beneath the overlay.
        rgb=float3(0,0.8,1)*0.12f+rgb*0.88f; transmittance*=0.88f;
    }
    output.write(float4(rgb,1-transmittance),pixel);
}

// Bounded surrogate: 3x3 isolated-splat patches, black background, fixed sample
// coordinates per step. Not the derivative of the full occlusion-aware composite.
// Each thread owns one splat, so no float atomics or concurrent parameter writes.
kernel void optimizeGaussians(device Gaussian* splats [[buffer(0)]],
                              constant Camera& camera [[buffer(1)]],
                              device const float* rgb [[buffer(2)]],
                              constant Settings& settings [[buffer(3)]], uint lane [[thread_position_in_grid]]) {
    if(lane>=min(settings.count,32u)) return;
    uint id=(settings.offset+lane)%settings.count;
    Gaussian g=splats[id]; D p[14],base[6];
    for(uint i=0;i<14;i++) p[i]=D(g.p[i]);
    if(!project(p,camera,base)) return;
    if(base[0].v<1 || base[1].v<1 || base[0].v>=camera.p[16]-1 || base[1].v>=camera.p[17]-1) return;
    float gradients[14];
    for(uint parameter=0;parameter<14;parameter++) {
        for(uint i=0;i<14;i++) p[i]=D(g.p[i],i==parameter?1:0);
        D c[6]; if(!project(p,camera,c)) return;
        D loss(0),det=c[3]*c[5]-c[4]*c[4];
        for(int y=-1;y<=1;y++) for(int x=-1;x<=1;x++) {
            float u=floor(base[0].v)+x,v=floor(base[1].v)+y;
            D dx=D(u)-c[0],dy=D(v)-c[1];
            D q=(c[5]*dx*dx-D(2)*c[4]*dx*dy+c[3]*dy*dy)/det;
            D alpha=dexp(-D(0.5)*q)/(D(1)+dexp(-p[10]));
            if(alpha.v>=0.99f) alpha=D(0.99f);
            uint pixel=uint(v)*uint(camera.p[16])+uint(u);
            for(uint k=0;k<3;k++) {
                D e=alpha*p[11+k]-D(rgb[pixel*3+k]); loss=loss+D(0.5)*e*e;
            }
        }
        gradients[parameter]=loss.d/9;
        if(!isfinite(gradients[parameter])) return;
    }
    for(uint i=0;i<14;i++) g.p[i]-=0.001f*clamp(gradients[i],-1.f,1.f);
    for(uint i=3;i<6;i++) g.p[i]=clamp(g.p[i],log(0.001f),log(0.5f));
    float norm=0; for(uint i=6;i<10;i++) norm+=g.p[i]*g.p[i];
    if(norm<1e-12f || !isfinite(norm)) return;
    for(uint i=6;i<10;i++) g.p[i]/=sqrt(norm);
    g.p[10]=clamp(g.p[10],-8.f,8.f);
    for(uint i=11;i<14;i++) g.p[i]=clamp(g.p[i],0.f,1.f);
    splats[id]=g;
}
