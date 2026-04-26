// ========== THREE.JS WEBGL VISUALIZER — REALISTIC INDIAN HIGH COURT ==========
const threeScript = document.createElement('script');
threeScript.src = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js";
threeScript.onload = function() {
    const container = document.getElementById('canvas-container');
    if (!container) return;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 0.85;
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x080a0f);
    scene.fog = new THREE.Fog(0x080a0f, 22, 40);

    const camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 0.1, 80);
    camera.position.set(0, 7, 16);

    // Materials
    const M = {
        teak:    new THREE.MeshStandardMaterial({ color: 0x3d1f0a, roughness: 0.55, metalness: 0.05 }),
        darkTeak:new THREE.MeshStandardMaterial({ color: 0x2a1205, roughness: 0.60, metalness: 0.00 }),
        stone:   new THREE.MeshStandardMaterial({ color: 0xd6ccbb, roughness: 0.85, metalness: 0.00 }),
        green:   new THREE.MeshStandardMaterial({ color: 0x1a3a2a, roughness: 0.90, metalness: 0.00 }),
        gold:    new THREE.MeshStandardMaterial({ color: 0xc9a84c, roughness: 0.30, metalness: 0.80 }),
        white:   new THREE.MeshStandardMaterial({ color: 0xf8f6f0, roughness: 0.80, metalness: 0.00 }),
        black:   new THREE.MeshStandardMaterial({ color: 0x111111, roughness: 0.60, metalness: 0.05 }),
        maroon:  new THREE.MeshStandardMaterial({ color: 0x6b1a1a, roughness: 0.50, metalness: 0.10 }),
        suit:    new THREE.MeshStandardMaterial({ color: 0x2c3e50, roughness: 0.65, metalness: 0.05 }),
        skin:    new THREE.MeshStandardMaterial({ color: 0xc68642, roughness: 0.65, metalness: 0.00 }),
        metal:   new THREE.MeshStandardMaterial({ color: 0x888888, roughness: 0.30, metalness: 0.90 }),
        cream:   new THREE.MeshStandardMaterial({ color: 0xf0e8d0, roughness: 0.90, metalness: 0.00 }),
    };
    function add(mesh) { scene.add(mesh); return mesh; }
    function box(w,h,d,mat,x,y,z) { const m=new THREE.Mesh(new THREE.BoxGeometry(w,h,d),mat); m.position.set(x,y,z); m.castShadow=true; m.receiveShadow=true; return add(m); }
    function cyl(rt,rb,h,seg,mat,x,y,z) { const m=new THREE.Mesh(new THREE.CylinderGeometry(rt,rb,h,seg),mat); m.position.set(x,y,z); m.castShadow=true; return add(m); }
    function sph(r,mat,x,y,z) { const m=new THREE.Mesh(new THREE.SphereGeometry(r,14,12),mat.clone()); m.position.set(x,y,z); return add(m); }

    // ── MARBLE FLOOR ──
    const cv=document.createElement('canvas'); cv.width=cv.height=512;
    const ctx2=cv.getContext('2d');
    for(let r=0;r<16;r++) for(let c=0;c<16;c++) {
        ctx2.fillStyle=(r+c)%2===0?'#e8e4dc':'#242018';
        ctx2.fillRect(c*32,r*32,32,32);
    }
    const tex=new THREE.CanvasTexture(cv); tex.wrapS=tex.wrapT=THREE.RepeatWrapping; tex.repeat.set(2,2);
    const floorMat=new THREE.MeshStandardMaterial({map:tex,roughness:0.2,metalness:0.1});
    const floor=new THREE.Mesh(new THREE.PlaneGeometry(34,26),floorMat);
    floor.rotation.x=-Math.PI/2; floor.receiveShadow=true; scene.add(floor);

    // ── ROOM SHELL ──
    box(34,0.3,26, new THREE.MeshStandardMaterial({color:0x1a0f05,roughness:0.9}), 0,10,0); // ceiling
    box(34,10,0.3, M.green, 0,5,-13);  // back wall
    box(0.3,10,26,  M.green,-17,5,0);  // left wall
    box(0.3,10,26,  M.green, 17,5,0);  // right wall
    box(34,1.5,0.2, M.darkTeak, 0,0.75,-12.85); // wainscot dado

    // ── COLUMNS (6, 3 per side) ──
    [-9,-1,7].forEach(z => {
        [-1,1].forEach(s => {
            cyl(0.35,0.4,9.5,16,M.stone, s*14,4.75,z);
            box(1.1,0.5,1.1, M.stone, s*14,9.75,z);
        });
    });

    // ── WINDOWS (glowing panels) ──
    const winMat=new THREE.MeshStandardMaterial({color:0x87ceeb,emissive:0x6ab8e8,emissiveIntensity:0.5,transparent:true,opacity:0.3,side:THREE.DoubleSide});
    [-8,0,8].forEach(z => [-1,1].forEach(s => {
        const w=new THREE.Mesh(new THREE.PlaneGeometry(3.5,4.5),winMat);
        w.position.set(s*16.8,5.5,z); w.rotation.y=s*Math.PI/2; scene.add(w);
    }));

    // ── JUDGE DAIS ──
    box(9,0.5,4,   M.stone, 0,0.25,-9);    // platform
    box(8,1.0,1.8, M.teak,  0,1.0,-9.2);   // desk top
    box(8,1.6,0.12,M.darkTeak,0,0.8,-8.3); // front panel
    box(1.3,2.4,0.28,M.maroon, 0,2.2,-10.1);// chair back
    box(1.3,0.15,1.1,M.maroon, 0,1.12,-9.7);// chair seat
    // National Emblem above bench
    cyl(0.28,0.38,0.9,8,  M.gold, 0,8.2,-12.8);
    add(Object.assign(new THREE.Mesh(new THREE.ConeGeometry(0.32,0.6,4),M.gold),{position:new THREE.Vector3(0,8.85,-12.8)}));
    box(4,0.65,0.08, M.gold, 0,7.4,-12.85); // Satyamev plaque

    // ── BAR TABLES ──
    function barTable(x,z) {
        box(5,0.1,1.4,M.teak,x,1.05,z);
        [[-2.2,-0.6],[2.2,-0.6],[-2.2,0.6],[2.2,0.6]].forEach(([dx,dz])=>box(0.12,1.05,0.12,M.darkTeak,x+dx,0.52,z+dz));
        // Mic
        cyl(0.02,0.02,0.6,6,M.metal,x,1.42,z-0.35);
        sph(0.065,new THREE.MeshStandardMaterial({color:0x333,roughness:0.4,metalness:0.8}),x,1.74,z-0.35);
        // Docs
        for(let i=0;i<5;i++) box(0.55,0.025,0.38,new THREE.MeshStandardMaterial({color:i%2?0xe8e0d0:0xf5f0e8}),x+1.5,1.12+i*0.028,z);
        // Name placard
        box(1.2,0.06,0.25,M.gold,x,1.12,z-0.55);
    }
    barTable(-5,-5);
    barTable(5,-5);

    // Reader desk
    box(2.2,0.1,1.2,M.teak, 4.8,1.05,-9);

    // ── WITNESS BOX ──
    box(2.4,0.22,2.4,M.stone, -8,0.11,-6.5);
    box(2.2,0.1,0.4,M.teak,  -8,1.18,-7.2);
    box(2.4,0.06,0.07,M.darkTeak,-8,1.05,-5.4); // front rail
    box(2.4,0.06,0.07,M.darkTeak,-8,1.05,-7.6);
    box(0.07,1.05,2.4,M.darkTeak,-6.9,0.63,-6.5);
    box(0.07,1.05,2.4,M.darkTeak,-9.1,0.63,-6.5);

    // ── PUBLIC GALLERY ──
    for(let r=0;r<3;r++) box(14,0.12,0.9,M.teak,0,0.48+r*0.35,4.5+r*1.3);
    box(14,0.9,0.08,M.darkTeak,0,0.45,3.2); // divider rail

    // ── TRICOLOR FLAG ──
    cyl(0.04,0.04,3.8,8,M.metal,13,1.9,10.5);
    [[0xFF9933,'saffron'],[0xffffff,'white'],[0x138808,'green']].forEach(([c],i)=>{
        const strip=new THREE.Mesh(new THREE.PlaneGeometry(1.7,0.36),new THREE.MeshStandardMaterial({color:c,side:THREE.DoubleSide}));
        strip.position.set(13.85,3.5-i*0.37,10.5); scene.add(strip);
    });
    const chakra=new THREE.Mesh(new THREE.TorusGeometry(0.13,0.025,8,24),new THREE.MeshStandardMaterial({color:0x000080,metalness:0.5}));
    chakra.position.set(13.85,3.13,10.49); scene.add(chakra);

    // ── HUMANOID AVATARS ──
    const agents = {};
    function humanoid(name,x,y,z,robeMat,isJudge) {
        const g=new THREE.Group();
        const rh=isJudge;
        // robe body
        const body=new THREE.Mesh(new THREE.CylinderGeometry(rh?0.33:0.27,rh?0.38:0.32,rh?1.7:1.5,10),robeMat.clone());
        body.position.y=rh?2.4:1.9; g.add(body);
        // shoulders + arms
        [-1,1].forEach(s=>{
            const sh=new THREE.Mesh(new THREE.SphereGeometry(rh?0.21:0.17,10,8),robeMat.clone());
            sh.position.set(s*(rh?0.4:0.32),rh?2.95:2.4,0); g.add(sh);
            const arm=new THREE.Mesh(new THREE.CylinderGeometry(0.07,0.06,0.8,8),robeMat.clone());
            arm.position.set(s*(rh?0.55:0.46),rh?2.55:2.05,0); arm.rotation.z=s*0.28; g.add(arm);
        });
        // white neckband
        const band=new THREE.Mesh(new THREE.CylinderGeometry(0.11,0.11,0.07,10),M.white);
        band.position.y=rh?3.2:2.68; g.add(band);
        // neck
        const neck=new THREE.Mesh(new THREE.CylinderGeometry(0.09,0.09,0.24,10),M.skin.clone());
        neck.position.y=rh?3.31:2.8; g.add(neck);
        // head (this is the glow mesh)
        const headMat=M.skin.clone();
        headMat.emissive=new THREE.Color(0,0,0);
        const head=new THREE.Mesh(new THREE.SphereGeometry(rh?0.25:0.21,14,12),headMat);
        head.position.y=rh?3.55:3.02; g.add(head);
        // judge wig
        if(rh){
            const wig=new THREE.Mesh(new THREE.SphereGeometry(0.28,14,12),M.cream.clone());
            wig.position.set(0,3.62,0); wig.scale.set(1,0.65,1); g.add(wig);
        }
        g.position.set(x,y,z); scene.add(g);
        agents[name]={group:g,glowMesh:head};
    }

    humanoid('judge',         0,    0.5,-9.5, M.maroon, true);
    humanoid('prosecutor',   -5,    0,  -6.5, M.black,  false);
    humanoid('defense',       5,    0,  -6.5, M.black,  false);
    humanoid('expert_witness',-8,   0,  -7,   M.suit,   false);
    humanoid('clerk',         4.8,  0,  -9.5, M.white,  false);

    // ── LIGHTING ──
    scene.add(new THREE.AmbientLight(0xfff5e0, 0.42));

    const sun=new THREE.DirectionalLight(0xfff0d0,0.6);
    sun.position.set(10,8,2); sun.castShadow=true; scene.add(sun);

    function spot(col,int,dist,angle,x,y,z,tx,ty,tz) {
        const s=new THREE.SpotLight(col,int,dist,angle,0.45);
        s.position.set(x,y,z); s.target.position.set(tx,ty,tz);
        s.castShadow=true; scene.add(s); scene.add(s.target); return s;
    }
    spot(0xffd700,3.5,20,Math.PI/7,  0,9.5,-7.5, 0,1.5,-9.5);    // judge gold
    spot(0xffffff,2.0,14,Math.PI/6, -6,8.5,-2,  -5,1,-6.5);       // prosecution
    spot(0xffffff,2.0,14,Math.PI/6,  6,8.5,-2,   5,1,-6.5);       // defense
    spot(0xe8f4ff,1.6,10,Math.PI/7, -8,7,  -3,  -8,1,-7);         // witness

    // Ceiling point lights
    [-9,-4,0,5,10].forEach(z=>{ const pl=new THREE.PointLight(0xffeedd,0.55,9); pl.position.set(0,9.5,z); scene.add(pl); });

    // Dust motes
    const dustGeo=new THREE.BufferGeometry();
    const DC=80; const dp=new Float32Array(DC*3);
    for(let i=0;i<DC;i++){ dp[i*3]=(Math.random()-.5)*18; dp[i*3+1]=Math.random()*8+0.5; dp[i*3+2]=(Math.random()-.5)*18-3; }
    dustGeo.setAttribute('position',new THREE.BufferAttribute(dp,3));
    scene.add(new THREE.Points(dustGeo,new THREE.PointsMaterial({color:0xffeebb,size:0.04,transparent:true,opacity:0.45})));

    // ── CAMERA VIEWS ──
    let activeSpeaker='none';
    let targetCamPos=new THREE.Vector3(0,7,16);
    let targetLookAt=new THREE.Vector3(0,2,-4);
    let currentLookAt=new THREE.Vector3(0,2,-4);
    const VIEWS={
        'none':          {pos:new THREE.Vector3(0,7,16),  lookAt:new THREE.Vector3(0,2,-4)},
        'judge':         {pos:new THREE.Vector3(0,4,0),   lookAt:new THREE.Vector3(0,3.5,-9.5)},
        'prosecutor':    {pos:new THREE.Vector3(-2,3.5,0),lookAt:new THREE.Vector3(-5,2,-6.5)},
        'defense':       {pos:new THREE.Vector3(2,3.5,0), lookAt:new THREE.Vector3(5,2,-6.5)},
        'expert_witness':{pos:new THREE.Vector3(-4,3,-3), lookAt:new THREE.Vector3(-8,2,-7)},
        'clerk':         {pos:new THREE.Vector3(3,4,-5),  lookAt:new THREE.Vector3(4.8,2,-9.5)}
    };

    // ── ANIMATE ──
    const clock=new THREE.Clock();
    function animate(){
        requestAnimationFrame(animate);
        const t=clock.getElapsedTime();
        // dust drift
        for(let i=0;i<DC;i++) dp[i*3+1]+=Math.sin(t*0.3+i)*0.0006;
        dustGeo.attributes.position.needsUpdate=true;
        // camera
        camera.position.lerp(targetCamPos,0.035);
        currentLookAt.lerp(targetLookAt,0.035);
        camera.lookAt(currentLookAt);
        // speaker pulse + bubble
        const bub=document.getElementById('floating-bubble');
        let bActive=false;
        for(const[name,data] of Object.entries(agents)){
            if(name===activeSpeaker&&data.glowMesh){
                data.glowMesh.material.emissive.setHex(0xffd700);
                data.glowMesh.material.emissiveIntensity=0.3+Math.sin(t*4)*0.3;
                const wp=new THREE.Vector3(); data.glowMesh.getWorldPosition(wp); wp.y+=0.5;
                const pr=wp.clone().project(camera);
                const sx=(pr.x*.5+.5)*container.clientWidth;
                const sy=(pr.y*-.5+.5)*container.clientHeight;
                if(pr.z<1&&bub){bub.style.left=sx+'px';bub.style.top=sy+'px';bActive=true;}
            } else if(data.glowMesh){
                data.glowMesh.material.emissiveIntensity=0;
            }
        }
        if(bub) bub.style.display=bActive?'block':'none';
        renderer.render(scene,camera);
    }
    animate();

    window.update3DState=function(speaker,text){
        activeSpeaker=speaker.toLowerCase();
        const v=VIEWS[activeSpeaker]||VIEWS['none'];
        targetCamPos=v.pos; targetLookAt=v.lookAt;
        const bn=document.getElementById('bubble-name');
        const bt=document.getElementById('bubble-text');
        if(bn&&bt&&activeSpeaker!=='none'){
            bn.innerText=activeSpeaker.replace(/_/g,' ').toUpperCase();
            bt.innerText=text;
        }
    };

    window.addEventListener('resize',()=>{
        camera.aspect=container.clientWidth/container.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth,container.clientHeight);
    });
};
document.head.appendChild(threeScript);
