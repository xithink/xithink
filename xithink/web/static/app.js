const messages = document.getElementById('messages');
const input = document.getElementById('messageInput');
const form = document.getElementById('composer');
const micButton = document.getElementById('micButton');
const liveTranscript = document.getElementById('liveTranscript');
const socketStatus = document.getElementById('socketStatus');
const semanticJson = document.getElementById('semanticJson');
const thoughtTrace = document.getElementById('thoughtTrace');
const cognitionJson = document.getElementById('cognitionJson');
const proofJson = document.getElementById('proofJson');
const autoSpeak = document.getElementById('autoSpeak');
const continuousVoice = document.getElementById('continuousVoice');
const cycles = document.getElementById('cycles');
const cyclesValue = document.getElementById('cyclesValue');
const environmentInput = document.getElementById('environment');
const worldviewJson = document.getElementById('worldviewJson');
const evolutionJson = document.getElementById('evolutionJson');
const growthJson = document.getElementById('growthJson');
const profileSelect = document.getElementById('profile');
const simExperiences = document.getElementById('simExperiences');
const simMaxAge = document.getElementById('simMaxAge');
const simSeed = document.getElementById('simSeed');
const simSkepticism = document.getElementById('simSkepticism');
const simSkepticismValue = document.getElementById('simSkepticismValue');
const simExploration = document.getElementById('simExploration');
const simExplorationValue = document.getElementById('simExplorationValue');
const simTesting = document.getElementById('simTesting');
const simTestingValue = document.getElementById('simTestingValue');
const runSimulation = document.getElementById('runSimulation');

let ws;
let recognition = null;
let listening = false;
let shouldRestartRecognition = false;
let interimText = '';

function connect() {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${protocol}//${location.host}/ws/chat`);
  ws.onopen = () => { socketStatus.textContent = '已连接'; };
  ws.onclose = () => {
    socketStatus.textContent = '连接断开，正在重连…';
    setTimeout(connect, 1200);
  };
  ws.onerror = () => { socketStatus.textContent = '连接异常'; };
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'ready') socketStatus.textContent = '已连接';
    if (data.type === 'thinking') socketStatus.textContent = '思考中…';
    if (data.type === 'semantic') {
      semanticJson.textContent = JSON.stringify(data.data, null, 2);
    }
    if (data.type === 'symbolic_answer') {
      proofJson.textContent = JSON.stringify(data.data, null, 2);
    }
    if (data.type === 'cognition') {
      cognitionJson.textContent = JSON.stringify(data.data, null, 2);
      worldviewJson.textContent = JSON.stringify({
        worldview: data.data.worldview,
        experience_rule_learning: data.data.experience_rule_learning,
        personal_symbols: data.data.personal_symbols,
        perspective_resonance: data.data.perspective_resonance,
      }, null, 2);
      evolutionJson.textContent = JSON.stringify({
        rule_evolution: data.data.rule_evolution,
        multi_perspective: data.data.multi_perspective,
      }, null, 2);
      growthJson.textContent = JSON.stringify({
        rule_genealogy: data.data.rule_genealogy,
        active_hypothesis_testing: data.data.active_hypothesis_testing,
        cognitive_profile: data.data.cognitive_profile,
      }, null, 2);
    }
    if (data.type === 'thought') addThought(data.data);
    if (data.type === 'reply') {
      socketStatus.textContent = '已连接';
      addMessage('assistant', data.text);
      if (autoSpeak.checked) speak(data.text);
      if (continuousVoice.checked && shouldRestartRecognition) restartRecognitionSoon();
    }
    if (data.type === 'error') addMessage('assistant', `错误：${data.message}`);
  };
}

function addMessage(role, text) {
  const article = document.createElement('article');
  article.className = `message ${role}`;
  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = role === 'user' ? '你' : 'X';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  article.append(avatar, bubble);
  messages.appendChild(article);
  messages.scrollTop = messages.scrollHeight;
}

function addThought(thought) {
  if (thoughtTrace.textContent === '等待思考…') thoughtTrace.textContent = '';
  const item = document.createElement('div');
  item.className = 'trace-item';
  const meta = document.createElement('small');
  meta.textContent = `cycle ${thought.cycle} · ${thought.kind} · confidence ${Number(thought.confidence).toFixed(3)}`;
  const content = document.createElement('div');
  content.textContent = thought.content;
  item.append(meta, content);
  thoughtTrace.prepend(item);
}

function sendText(text) {
  text = text.trim();
  if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;
  addMessage('user', text);
  thoughtTrace.textContent = '等待思考…';
  ws.send(JSON.stringify({ text, cycles: Number(cycles.value), environment: environmentInput.value.trim() || null, profile: profileSelect.value }));
  input.value = '';
}

form.addEventListener('submit', (event) => {
  event.preventDefault();
  sendText(input.value);
});

input.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    sendText(input.value);
  }
});

cycles.addEventListener('input', () => { cyclesValue.textContent = cycles.value; });

document.querySelectorAll('.tab').forEach((tab) => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(x => x.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById(`${tab.dataset.tab}Panel`).classList.add('active');
  });
});

function setupSpeechRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    micButton.disabled = true;
    micButton.title = '当前浏览器不支持 Web Speech Recognition；建议使用新版 Chrome/Edge';
    liveTranscript.textContent = '当前浏览器不支持浏览器端语音识别，可继续使用文字对话。';
    return;
  }
  recognition = new SpeechRecognition();
  recognition.lang = 'zh-CN';
  recognition.continuous = true;
  recognition.interimResults = true;

  recognition.onstart = () => {
    listening = true;
    micButton.classList.add('listening');
    liveTranscript.textContent = '正在聆听…';
  };

  recognition.onresult = (event) => {
    let finalText = '';
    interimText = '';
    for (let i = event.resultIndex; i < event.results.length; i += 1) {
      const transcript = event.results[i][0].transcript;
      if (event.results[i].isFinal) finalText += transcript;
      else interimText += transcript;
    }
    liveTranscript.textContent = interimText || '正在聆听…';
    if (finalText.trim()) {
      liveTranscript.textContent = `识别完成：${finalText.trim()}`;
      sendText(finalText.trim());
      if (!continuousVoice.checked) stopRecognition();
    }
  };

  recognition.onerror = (event) => {
    if (event.error !== 'no-speech' && event.error !== 'aborted') {
      liveTranscript.textContent = `语音识别错误：${event.error}`;
    }
  };

  recognition.onend = () => {
    listening = false;
    micButton.classList.remove('listening');
    if (shouldRestartRecognition && continuousVoice.checked && !speechSynthesis.speaking) {
      restartRecognitionSoon();
    }
  };
}

function startRecognition() {
  if (!recognition || listening) return;
  shouldRestartRecognition = true;
  try { recognition.start(); } catch (_) {}
}

function stopRecognition() {
  shouldRestartRecognition = false;
  if (recognition && listening) recognition.stop();
  liveTranscript.textContent = '语音模式已暂停';
}

function restartRecognitionSoon() {
  setTimeout(() => {
    if (shouldRestartRecognition && !listening && !speechSynthesis.speaking) startRecognition();
  }, 350);
}

micButton.addEventListener('click', () => {
  if (listening || shouldRestartRecognition) stopRecognition();
  else startRecognition();
});

function speak(text) {
  if (!('speechSynthesis' in window)) return;
  if (recognition && listening) recognition.stop();
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = 'zh-CN';
  utterance.rate = 1.0;
  utterance.onend = () => {
    if (continuousVoice.checked && shouldRestartRecognition) restartRecognitionSoon();
  };
  window.speechSynthesis.speak(utterance);
}

setupSpeechRecognition();
connect();


[simSkepticism, simExploration, simTesting].forEach((el) => {
  el.addEventListener('input', () => {
    simSkepticismValue.textContent = Number(simSkepticism.value).toFixed(2);
    simExplorationValue.textContent = Number(simExploration.value).toFixed(2);
    simTestingValue.textContent = Number(simTesting.value).toFixed(2);
  });
});

runSimulation.addEventListener('click', async () => {
  runSimulation.disabled = true;
  runSimulation.textContent = '模拟中…';
  growthJson.textContent = '正在运行年龄/经历模拟…';
  try {
    const response = await fetch('/api/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        profile: profileSelect.value,
        total_experiences: Number(simExperiences.value),
        max_age: Number(simMaxAge.value),
        seed: Number(simSeed.value),
        skepticism: Number(simSkepticism.value),
        exploration: Number(simExploration.value),
        active_test_aggressiveness: Number(simTesting.value),
      }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'simulation failed');
    growthJson.textContent = JSON.stringify(data, null, 2);
  } catch (error) {
    growthJson.textContent = `模拟失败：${error.message}`;
  } finally {
    runSimulation.disabled = false;
    runSimulation.textContent = '运行年龄模拟';
  }
});
