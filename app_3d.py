import gradio as gr
import time

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HTML / JS / WebGL 3D Bridge (Three.js)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THREE_JS_BRIDGE = """
<div id="visualizer-wrapper" style="position: relative; width: 100%; height: 500px; border-radius: 12px; overflow: hidden; background: #050510;">
    <!-- 3D Canvas Container -->
    <div id="canvas-container" style="width: 100%; height: 100%;"></div>
    
    <!-- Floating Text Bubble for Speaking Agent -->
    <div id="floating-bubble" style="position: absolute; display: none; background: rgba(10,10,25,0.9); border: 1px solid rgba(99,102,241,0.5); color: #e2e8f0; padding: 12px; border-radius: 8px; font-family: 'Inter', sans-serif; font-size: 14px; max-width: 280px; z-index: 10; transform: translate(-50%, -100%); pointer-events: none; box-shadow: 0 4px 20px rgba(0,0,0,0.5);">
        <strong id="bubble-name" style="color: #fbbf24; display: block; margin-bottom: 4px;">Speaker</strong>
        <span id="bubble-text">Argument text...</span>
    </div>
</div>

<!-- Load Three.js -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>

<script>
(function() {
    // ── 1. WebGL Scene Setup ──
    const container = document.getElementById('canvas-container');
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x050510);
    scene.fog = new THREE.FogExp2(0x050510, 0.04);

    const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 100);
    camera.position.set(0, 5, 10);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    // ── 2. Lighting ──
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.3);
    scene.add(ambientLight);
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.6);
    dirLight.position.set(5, 10, 5);
    scene.add(dirLight);

    // ── 3. Courtroom Geometry (Low-Poly Stylized) ──
    const agents = {};

    function createAgent(name, x, y, z, colorStr, isJudge=false) {
        const color = new THREE.Color(colorStr);
        // Desk
        const deskGeo = new THREE.BoxGeometry(isJudge ? 4 : 2, isJudge ? 1.5 : 1, 1);
        const deskMat = new THREE.MeshLambertMaterial({ color: 0x1a1a2e });
        const desk = new THREE.Mesh(deskGeo, deskMat);
        desk.position.set(x, y - (isJudge ? 0.75 : 0.5), z + (isJudge ? 1 : -0.5));
        scene.add(desk);

        // Avatar (Primitive Cylinder)
        const geo = new THREE.CylinderGeometry(0.4, 0.5, 1.2, 16);
        const mat = new THREE.MeshStandardMaterial({ 
            color: color, 
            roughness: 0.2, 
            metalness: 0.8,
            emissive: color,
            emissiveIntensity: 0.0 
        });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.set(x, y, z);
        scene.add(mesh);

        agents[name] = { mesh: mesh, baseColor: color };
    }

    // Floor
    const floorGeo = new THREE.PlaneGeometry(30, 30);
    const floorMat = new THREE.MeshStandardMaterial({ color: 0x0a0a15, roughness: 0.8 });
    const floor = new THREE.Mesh(floorGeo, floorMat);
    floor.rotation.x = -Math.PI / 2;
    scene.add(floor);

    // Placements
    createAgent('judge', 0, 2, -4, '#fbbf24', true);       // Gold, Elevated Center
    createAgent('prosecutor', -3, 1, 0, '#ef4444', false); // Red, Left
    createAgent('defense', 3, 1, 0, '#22c55e', false);     // Green, Right

    // ── 4. Camera & Animation States ──
    let activeSpeaker = 'none';
    let targetCamPos = new THREE.Vector3(0, 6, 12);
    let targetLookAt = new THREE.Vector3(0, 1, -2);
    let currentLookAt = new THREE.Vector3(0, 1, -2);

    const VIEWS = {
        'none':       { pos: new THREE.Vector3(0, 6, 12), lookAt: new THREE.Vector3(0, 1, -2) },
        'judge':      { pos: new THREE.Vector3(0, 3, 2),  lookAt: new THREE.Vector3(0, 2, -4) },
        'prosecutor': { pos: new THREE.Vector3(-1, 2, 4), lookAt: new THREE.Vector3(-3, 1, 0) },
        'defense':    { pos: new THREE.Vector3(1, 2, 4),  lookAt: new THREE.Vector3(3, 1, 0) }
    };

    // ── 5. Render Loop ──
    const clock = new THREE.Clock();

    function animate() {
        requestAnimationFrame(animate);
        const time = clock.getElapsedTime();

        // Smooth Camera Lerp
        camera.position.lerp(targetCamPos, 0.05);
        currentLookAt.lerp(targetLookAt, 0.05);
        camera.lookAt(currentLookAt);

        // Pulsing Aura & Floating Bubble positioning
        const bubble = document.getElementById('floating-bubble');
        let bubbleActive = false;

        for (const [name, data] of Object.entries(agents)) {
            if (name === activeSpeaker) {
                // Pulse emissive intensity
                data.mesh.material.emissiveIntensity = 0.5 + Math.sin(time * 5) * 0.5;
                
                // Project 3D coordinate to 2D screen space for bubble
                const vec = data.mesh.position.clone();
                vec.y += 1.2; // Hover above head
                vec.project(camera);
                
                const x = (vec.x * .5 + .5) * container.clientWidth;
                const y = (vec.y * -.5 + .5) * container.clientHeight;
                
                if (vec.z < 1) { // Only show if in front of camera
                    bubble.style.left = `${x}px`;
                    bubble.style.top = `${y}px`;
                    bubbleActive = true;
                }
            } else {
                data.mesh.material.emissiveIntensity = 0.0;
            }
        }

        bubble.style.display = bubbleActive ? 'block' : 'none';
        renderer.render(scene, camera);
    }
    animate();

    // ── 6. Gradio-to-JS Bridge (Hidden State Polling) ──
    // Gradio uses Textareas inside the component divs
    let lastSpeaker = null;
    let lastText = null;

    setInterval(() => {
        // Find the hidden inputs by their Gradio elem_id
        const speakerEl = document.querySelector('#hidden_speaker textarea');
        const textEl = document.querySelector('#hidden_text textarea');

        if (speakerEl && textEl) {
            const currentSpk = speakerEl.value.trim().toLowerCase();
            const currentTxt = textEl.value.trim();

            if (currentSpk !== lastSpeaker || currentTxt !== lastText) {
                lastSpeaker = currentSpk;
                lastText = currentTxt;

                // Update 3D State
                activeSpeaker = currentSpk;
                if (VIEWS[activeSpeaker]) {
                    targetCamPos = VIEWS[activeSpeaker].pos;
                    targetLookAt = VIEWS[activeSpeaker].lookAt;
                } else {
                    targetCamPos = VIEWS['none'].pos;
                    targetLookAt = VIEWS['none'].lookAt;
                }

                // Update Bubble
                if (activeSpeaker !== 'none') {
                    document.getElementById('bubble-name').innerText = activeSpeaker.toUpperCase();
                    document.getElementById('bubble-text').innerText = currentTxt;
                }
            }
        }
    }, 200);

    // Handle Window Resize
    window.addEventListener('resize', () => {
        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, container.clientHeight);
    });
})();
</script>
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Gradio Application
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with gr.Blocks() as app:
    gr.Markdown("## 🏛️ AI Justice Arena — Reactive 3D WebGL Visualizer")
    gr.Markdown("Watch the camera dynamically pan and focus on the active RL agent.")
    
    with gr.Row():
        with gr.Column(scale=3):
            # The 3D Scene
            gr.HTML(THREE_JS_BRIDGE)
        
        with gr.Column(scale=1):
            # Python Simulation Loop Controls
            gr.Markdown("### Python Backend Controller")
            gr.Markdown("Simulate the Python RL loop picking an agent and generating text.")
            
            speaker_dropdown = gr.Dropdown(
                choices=["none", "judge", "prosecutor", "defense"], 
                value="none", 
                label="Active Agent"
            )
            text_input = gr.Textbox(
                lines=4, 
                value="Court is now in session.", 
                label="Generated Argument"
            )
            trigger_btn = gr.Button("Simulate Next Turn", variant="primary")
            
            # Hidden bridge states (These act as the sync mechanism between Python and JS)
            hidden_speaker = gr.Textbox(elem_id="hidden_speaker", visible=False)
            hidden_text = gr.Textbox(elem_id="hidden_text", visible=False)

    def trigger_turn(speaker, text):
        """Python function acting as the OpenEnv simulation step."""
        time.sleep(0.2) # Simulate LLM thinking time
        return speaker, text

    trigger_btn.click(
        fn=trigger_turn,
        inputs=[speaker_dropdown, text_input],
        outputs=[hidden_speaker, hidden_text]
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7861)
