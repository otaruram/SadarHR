import React, { useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import { Activity, AlertTriangle, CheckCircle, TrendingUp, User, BookOpen, ChevronLeft, ShieldCheck, Zap, ArrowRight, ArrowLeft, Lock, Clock, MapPin, Award } from 'lucide-react';

const GithubIcon = ({ className }: { className?: string }) => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <path d="M15 22v-4a4.8 4.8 0 0 0-1-3.2c3-.3 6-1.5 6-6.5a5.5 5.5 0 0 0-1.5-3.8 5.5 5.5 0 0 0-.1-3.8s-1.2-.4-3.9 1.4a13.3 13.3 0 0 0-7 0C6.2 2.7 5 3.1 5 3.1a5.5 5.5 0 0 0-.1 3.8A5.5 5.5 0 0 0 3 10.7c0 5 3 6.2 6 6.5a4.8 4.8 0 0 0-1 3.2v4"></path>
  </svg>
);

interface PreprocessMeta {
  features: string[];
  categorical_mapping: Record<string, Record<string, number>>;
  numerical_scaling: Record<string, { mean: number; std: number }>;
}

type ViewState = 'landing' | 'dashboard' | 'docs';

const STEP_1 = ['Age', 'Gender', 'MaritalStatus', 'EducationLevel', 'NumDependents', 'DistanceFromHome_km', 'Department', 'JobRole', 'JobLevel'];
const STEP_2 = ['YearsAtCompany', 'YearsInCurrentRole', 'YearsWithCurrentManager', 'YearsSinceLastPromotion', 'NumCompaniesWorked', 'TotalWorkingYears', 'BaseSalary_USD', 'StockOptionLevel', 'PercentSalaryHike', 'BonusPct', 'TrainingHoursLastYear', 'NumProjects', 'AbsenteeismDays', 'TravelFrequency', 'RemoteWorkPct'];
const STEP_3 = ['OverTime', 'JobSatisfaction', 'EnvironmentSatisfaction', 'RelationshipSatisfaction', 'WorkLifeBalance', 'ManagerRating', 'CompanyCultureScore', 'PerformanceRating', 'PeerReviewScore'];

const EMOTIVE_SCALES = [
  'JobSatisfaction', 'EnvironmentSatisfaction', 'RelationshipSatisfaction',
  'WorkLifeBalance', 'ManagerRating', 'CompanyCultureScore', 'PerformanceRating',
  'PeerReviewScore'
];

const svgFavicon = `data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M4 20v-6a1 1 0 0 1 1-1h2a1 1 0 0 1 1 1v6" fill="%23fca5a5"/><path d="M10 20V10a1 1 0 0 1 1-1h2a1 1 0 0 1 1 1v10" fill="%23fca5a5"/><path d="M21.5 7.5c0 2.8-2.54 5.1-6.38 8.52L14.9 16.2l-.22-.2C10.84 12.6 8.3 10.3 8.3 7.5c0-2.07 1.68-3.75 3.75-3.75 1.2 0 2.3.6 2.85 1.5.55-.9 1.65-1.5 2.85-1.5 2.07 0 3.75 1.68 3.75 3.75z" fill="%23dc2626"/></svg>`;

const SadarHRLogo = ({ className = "w-6 h-6" }: { className?: string }) => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" className={className}>
    <path d="M4 20v-6a1 1 0 0 1 1-1h2a1 1 0 0 1 1 1v6" fill="#fca5a5" />
    <path d="M10 20V10a1 1 0 0 1 1-1h2a1 1 0 0 1 1 1v10" fill="#fca5a5" />
    <path d="M21.5 7.5c0 2.8-2.54 5.1-6.38 8.52L14.9 16.2l-.22-.2C10.84 12.6 8.3 10.3 8.3 7.5c0-2.07 1.68-3.75 3.75-3.75 1.2 0 2.3.6 2.85 1.5.55-.9 1.65-1.5 2.85-1.5 2.07 0 3.75 1.68 3.75 3.75z" fill="currentColor" />
  </svg>
);

function App() {
  const [viewState, setViewState] = useState<ViewState>('landing');
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  const [meta, setMeta] = useState<PreprocessMeta | null>(null);
  const [loading, setLoading] = useState(true);
  
  const [formData, setFormData] = useState<Record<string, any>>({});
  const [prediction, setPrediction] = useState<number | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [currentStep, setCurrentStep] = useState<1 | 2 | 3>(1);
  
  useEffect(() => {
    // Inject dynamic SVG favicon
    let link = document.querySelector("link[rel~='icon']") as HTMLLinkElement;
    if (!link) {
      link = document.createElement('link');
      link.rel = 'icon';
      document.getElementsByTagName('head')[0].appendChild(link);
    }
    link.href = svgFavicon;

    async function loadModelAndMeta() {
      try {
        setLoading(true);
        const loadedModel = await tf.loadLayersModel('/model/model.json');
        setModel(loadedModel);
        
        const response = await fetch('/model/preprocess_meta.json');
        if (!response.ok) throw new Error('Failed to load preprocess metadata');
        const loadedMeta: PreprocessMeta = await response.json();
        setMeta(loadedMeta);
        
        const initialData: Record<string, any> = {};
        loadedMeta.features.forEach(feat => {
          initialData[feat] = '';
        });
        
        setFormData(initialData);
        
      } catch (err: any) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    }
    
    loadModelAndMeta();
  }, []);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    const isCategorical = meta?.categorical_mapping[name];
    const isEmotive = EMOTIVE_SCALES.includes(name);
    setFormData(prev => ({
      ...prev,
      [name]: isCategorical && !isEmotive ? value : Number(value) || value
    }));
    
    // Reset prediction when user modifies any input
    setPrediction(null);
  };

  const handlePredict = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!model || !meta) return;
    
    setIsPredicting(true);
    await new Promise(r => setTimeout(r, 800));
    
    try {
      const inputFeatures = meta.features.map(featureName => {
        let value = formData[featureName];
        if (meta.categorical_mapping[featureName]) {
          value = meta.categorical_mapping[featureName][String(value)] || 0;
        } else if (meta.numerical_scaling[featureName]) {
          const { mean, std } = meta.numerical_scaling[featureName];
          value = (Number(value) - mean) / std;
        }
        return value;
      });
      
      const inputTensor = tf.tensor2d([inputFeatures]);
      const resultTensor = model.predict(inputTensor) as tf.Tensor;
      const prob = (await resultTensor.data())[0];
      setPrediction(prob);
      
      setTimeout(() => {
        document.getElementById('result-panel')?.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }, 100);
      
      inputTensor.dispose();
      resultTensor.dispose();
    } catch (err: any) {
      console.error(err);
    } finally {
      setIsPredicting(false);
    }
  };

  const isStepValid = (step: 1 | 2 | 3) => {
    const features = step === 1 ? STEP_1 : step === 2 ? STEP_2 : STEP_3;
    return features.every(f => formData[f] !== undefined && formData[f] !== '');
  };

  if (viewState === 'docs') {
    return (
      <div className="min-h-screen bg-gray-50 text-gray-800 font-sans p-4 md:p-8">
        <div className="max-w-4xl mx-auto bg-white rounded-2xl md:rounded-3xl shadow-sm border border-gray-100 overflow-hidden">
          <header className="p-5 md:p-8 border-b border-gray-100 flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-red-600 text-white">
            <div className="flex items-center gap-3">
              <BookOpen className="w-6 h-6 md:w-8 md:h-8 flex-shrink-0" />
              <h1 className="text-xl md:text-2xl font-extrabold tracking-tight">Dokumentasi Teknis</h1>
            </div>
            <button 
              onClick={() => setViewState('landing')}
              className="flex items-center justify-center gap-1 px-4 py-2 text-sm font-bold text-red-600 bg-white rounded-full hover:bg-gray-100 transition-colors w-full sm:w-auto"
            >
              <ChevronLeft className="w-4 h-4" /> Kembali
            </button>
          </header>
          
          <div className="p-5 md:p-10 space-y-8 md:space-y-12">
            <section>
              <h2 className="text-lg md:text-xl font-bold text-gray-900 mb-3 md:mb-4 flex items-center gap-2">
                <Activity className="w-5 h-5 md:w-6 md:h-6 text-red-500" /> Pengenalan
              </h2>
              <p className="text-sm md:text-base text-gray-600 leading-relaxed">
                Platform <strong>SadarHR Attrition Predictor</strong> dirancang untuk membantu tim Sumber Daya Manusia (HR) dalam mengidentifikasi probabilitas seorang karyawan untuk resign (attrition) berdasarkan profil, kinerja, dan data historis.
              </p>
            </section>
            <section>
              <h2 className="text-lg md:text-xl font-bold text-gray-900 mb-3 md:mb-4 flex items-center gap-2">
                <Zap className="w-5 h-5 md:w-6 md:h-6 text-red-500" /> Arsitektur Serverless
              </h2>
              <div className="bg-red-50 p-5 md:p-6 rounded-2xl">
                <p className="text-sm md:text-base text-gray-700 leading-relaxed mb-4">
                  Aplikasi ini menggunakan pendekatan inovatif <strong>Serverless Machine Learning</strong>. Model Neural Network dilatih menggunakan Keras/TensorFlow dan dikonversi menjadi format TensorFlow.js.
                </p>
                <ul className="space-y-3">
                  <li className="flex items-start gap-3">
                    <CheckCircle className="w-4 h-4 md:w-5 md:h-5 text-red-500 mt-1 flex-shrink-0" />
                    <span className="text-sm md:text-base"><strong>Inferensi di Browser:</strong> Proses prediksi berjalan langsung di perangkat pengguna, membebaskan beban server backend.</span>
                  </li>
                  <li className="flex items-start gap-3">
                    <CheckCircle className="w-4 h-4 md:w-5 md:h-5 text-red-500 mt-1 flex-shrink-0" />
                    <span className="text-sm md:text-base"><strong>Kecepatan Maksimal:</strong> Prediksi selesai dalam hitungan milidetik tanpa latensi jaringan.</span>
                  </li>
                </ul>
              </div>
            </section>
            <section>
              <h2 className="text-lg md:text-xl font-bold text-gray-900 mb-3 md:mb-4 flex items-center gap-2">
                <ShieldCheck className="w-5 h-5 md:w-6 md:h-6 text-red-500" /> Keamanan & Privasi Data 100%
              </h2>
              <p className="text-sm md:text-base text-gray-600 leading-relaxed">
                Karena menggunakan TensorFlow.js di sisi klien (front-end), <strong>tidak ada satu pun data personal karyawan yang dikirimkan ke server atau API eksternal</strong>.
              </p>
            </section>
            <section>
              <h2 className="text-lg md:text-xl font-bold text-gray-900 mb-3 md:mb-4 flex items-center gap-2">
                <User className="w-5 h-5 md:w-6 md:h-6 text-red-500" /> Model Data & Variabel
              </h2>
              <p className="text-sm md:text-base text-gray-600 leading-relaxed mb-4">
                Model ini di-training dengan menggunakan <strong>34 parameter spesifik</strong> yang mewakili dinamika karyawan. Input secara otomatis ditangani dengan teknik pra-pemrosesan:
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="border border-gray-100 rounded-2xl p-4 md:p-5 shadow-sm">
                  <h4 className="font-bold text-gray-800 mb-2">Data Numerik</h4>
                  <p className="text-xs md:text-sm text-gray-500">Nilai kosong otomatis diisi menggunakan nilai <em>Median</em>, lalu distandarisasi menggunakan StandardScaler berdasarkan nilai mean/std di data latih.</p>
                </div>
                <div className="border border-gray-100 rounded-2xl p-4 md:p-5 shadow-sm">
                  <h4 className="font-bold text-gray-800 mb-2">Data Kategorikal</h4>
                  <p className="text-xs md:text-sm text-gray-500">Nilai kosong diisi dengan nilai yang sering muncul (<em>Mode</em>). Variabel teks diubah menjadi Label Encoding numerik.</p>
                </div>
              </div>
            </section>
          </div>
        </div>
      </div>
    );
  }

  if (viewState === 'landing') {
    return (
      <div className="min-h-screen bg-neutral-50 text-gray-900 font-sans flex flex-col">
        <nav className="w-full px-4 md:px-8 py-4 md:py-6 flex items-center justify-between border-b border-gray-200 bg-white shadow-sm">
          <div className="flex items-center gap-2">
            <SadarHRLogo className="w-8 h-8 md:w-10 md:h-10 text-red-600" />
            <span className="font-bold text-lg md:text-xl tracking-tight text-gray-800">Sadar<span className="text-red-600">HR</span></span>
          </div>
          <div className="flex items-center gap-3 md:gap-6">
            <a 
              href="https://github.com/otaruram/SadarHR" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-gray-400 hover:text-gray-900 transition-colors"
            >
              <GithubIcon className="w-5 h-5 md:w-6 md:h-6" />
            </a>
            <button 
              onClick={() => setViewState('docs')}
              className="text-xs md:text-sm font-semibold text-gray-500 hover:text-gray-900 transition-colors flex items-center gap-1.5"
            >
              <BookOpen className="w-3.5 h-3.5 md:w-4 md:h-4" />
              <span className="hidden sm:inline">Dokumentasi</span>
              <span className="inline sm:hidden">Docs</span>
            </button>
          </div>
        </nav>

        <main className="flex-1 flex flex-col items-center justify-center px-4 py-16 md:py-24 text-center max-w-5xl mx-auto">
          <h1 className="font-serif text-4xl md:text-6xl lg:text-7xl font-extrabold tracking-tight text-gray-900 leading-[1.1] mb-6 md:mb-8">
            Pahami polanya, jaga talentanya.<br />
            Deteksi <span className="text-red-600">risiko resign</span> sebelum<br />
            suratnya sampai di mejamu.
          </h1>
          <p className="text-base md:text-xl text-gray-600 mb-10 md:mb-12 max-w-3xl mx-auto leading-relaxed">
            SadarHR membantu mendeteksi risiko kejenuhan karyawan secara proaktif. Berjalan 100% serverless di browser Anda tanpa risiko kebocoran data perusahaan.
          </p>
          
          <div className="flex flex-col items-center w-full">
            <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-4 w-full sm:w-auto mb-6">
              <button
                onClick={() => setViewState('dashboard')}
                className="group relative px-8 py-4 bg-red-600 text-white rounded-full font-bold text-lg shadow-lg hover:shadow-xl hover:bg-red-700 transition-all hover:-translate-y-1"
              >
                <span className="flex items-center justify-center gap-2">
                  Buka Aplikasi
                  <TrendingUp className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                </span>
              </button>
              <button
                onClick={() => setViewState('docs')}
                className="px-8 py-4 bg-white text-gray-700 rounded-full font-bold text-lg border border-gray-300 hover:border-gray-400 hover:bg-gray-50 transition-all"
              >
                Dokumentasi
              </button>
            </div>
            
            <div className="flex items-center justify-center gap-2 text-xs md:text-sm text-gray-500">
              <Lock className="w-4 h-4 text-green-600" />
              <span>🔒 Data internal perusahaan Anda aman. Tidak ada transfer data keluar jaringan lokal Anda.</span>
            </div>
          </div>

          <div className="mt-20 md:mt-28 w-full text-left">
            <h2 className="text-2xl md:text-3xl font-bold font-serif text-gray-800 text-center mb-8 md:mb-12">Bagaimana SadarHR Membantu Anda?</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 md:gap-8">
              <div className="bg-white p-6 md:p-8 rounded-2xl md:rounded-3xl shadow-sm border border-gray-100 hover:shadow-md transition-shadow">
                <div className="w-12 h-12 rounded-xl bg-orange-50 flex items-center justify-center mb-5 text-orange-600">
                  <Clock className="w-6 h-6" />
                </div>
                <h3 className="text-lg font-bold text-gray-900 mb-3 font-serif">Deteksi Burnout Lembur</h3>
                <p className="text-gray-600 text-sm leading-relaxed">Karyawan performa tinggi, tapi sebulan terakhir Overtime (lembur) naik 40% dan Work-Life Balance drop. SadarHR langsung memberikan sinyal kuning.</p>
              </div>
              <div className="bg-white p-6 md:p-8 rounded-2xl md:rounded-3xl shadow-sm border border-gray-100 hover:shadow-md transition-shadow">
                <div className="w-12 h-12 rounded-xl bg-blue-50 flex items-center justify-center mb-5 text-blue-600">
                  <MapPin className="w-6 h-6" />
                </div>
                <h3 className="text-lg font-bold text-gray-900 mb-3 font-serif">Faktor Jarak & Logistik</h3>
                <p className="text-gray-600 text-sm leading-relaxed">Talenta terbaik baru pindah rumah yang jaraknya 50km dari kantor. SadarHR membantu mendeteksi risiko kejenuhan akibat perjalanan jauh.</p>
              </div>
              <div className="bg-white p-6 md:p-8 rounded-2xl md:rounded-3xl shadow-sm border border-gray-100 hover:shadow-md transition-shadow">
                <div className="w-12 h-12 rounded-xl bg-purple-50 flex items-center justify-center mb-5 text-purple-600">
                  <Award className="w-6 h-6" />
                </div>
                <h3 className="text-lg font-bold text-gray-900 mb-3 font-serif">Proteksi Top Performer</h3>
                <p className="text-gray-600 text-sm leading-relaxed">Memberikan alarm dini pada karyawan berkinerja tinggi yang sudah lama tidak mendapatkan promosi dan mulai merasa jenuh.</p>
              </div>
            </div>
          </div>

          <div className="mt-20 md:mt-24 w-full border-t border-gray-200 pt-10 pb-8">
            <h2 className="text-xl font-bold font-serif text-gray-800 text-center mb-6">Tim Pengembang</h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 max-w-2xl mx-auto">
              <a href="https://www.linkedin.com/in/otaruram/" target="_blank" rel="noopener noreferrer" className="flex items-center gap-4 bg-white p-4 rounded-2xl border border-gray-100 hover:border-red-200 hover:shadow-md transition-all group">
                <img src="/oki.png" alt="Oki Taruna Ramadhan" className="w-14 h-14 rounded-full object-cover object-top border-2 border-transparent group-hover:border-red-500 transition-colors shadow-sm shrink-0" />
                <div className="text-left flex-1">
                  <p className="font-bold text-gray-900 group-hover:text-red-600 transition-colors leading-tight mb-0.5">Oki Taruna Ramadhan</p>
                  <p className="text-xs text-gray-500">Lead</p>
                </div>
              </a>
              <a href="https://www.linkedin.com/in/azdzikri-muhammad-hisyam-soleh-086b43281/" target="_blank" rel="noopener noreferrer" className="flex items-center gap-4 bg-white p-4 rounded-2xl border border-gray-100 hover:border-red-200 hover:shadow-md transition-all group">
                <img src="/azikri.png" alt="Azdzikri Muhammad Hisyam Soleh" className="w-14 h-14 rounded-full object-cover object-[50%_25%] border-2 border-transparent group-hover:border-red-500 transition-colors shadow-sm shrink-0" />
                <div className="text-left flex-1">
                  <p className="font-bold text-gray-900 group-hover:text-red-600 transition-colors leading-tight mb-0.5">Azdzikri Muhammad Hisyam Soleh</p>
                  <p className="text-xs text-gray-500">Assistant Lead</p>
                </div>
              </a>
            </div>
          </div>
        </main>
        <footer className="w-full text-center text-gray-400 text-xs md:text-sm pb-8 bg-transparent">
          Made by Tim SadarHR. All rights reserved.
        </footer>
      </div>
    );
  }

  const currentFeatures = currentStep === 1 ? STEP_1 : currentStep === 2 ? STEP_2 : STEP_3;

  return (
    <div className="min-h-screen bg-gray-50 text-gray-800 font-sans p-3 md:p-8">
      <div className="max-w-6xl mx-auto">
        <header className="mb-4 md:mb-8 flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-4 bg-white p-4 md:p-6 rounded-2xl md:rounded-3xl shadow-sm border border-gray-100">
          <div className="flex items-center gap-3 md:gap-4">
            <SadarHRLogo className="w-10 h-10 md:w-12 md:h-12 text-red-600 drop-shadow-sm" />
            <div>
              <h1 className="text-lg md:text-2xl font-extrabold tracking-tight text-gray-900">
                Sadar<span className="text-red-600">HR</span> Predictor
              </h1>
              <p className="text-gray-500 text-xs md:text-sm">
                Screening Karyawan Internal HR
              </p>
            </div>
          </div>
          <button 
            onClick={() => setViewState('landing')}
            className="flex justify-center items-center gap-1.5 px-4 py-2 md:px-5 md:py-2.5 text-sm font-bold text-gray-600 bg-gray-100 rounded-lg md:rounded-xl hover:bg-gray-200 hover:text-gray-900 transition-colors w-full sm:w-auto"
          >
            <ChevronLeft className="w-4 h-4" /> <span className="hidden sm:inline">Kembali ke Awal</span>
          </button>
        </header>

        {loading ? (
          <div className="flex flex-col items-center justify-center py-20 bg-white rounded-2xl md:rounded-3xl shadow-sm border border-gray-100">
            <div className="animate-spin rounded-full h-10 w-10 md:h-12 md:w-12 border-b-2 border-red-600 mb-4"></div>
            <p className="text-gray-500 text-sm md:text-base">Memuat Model TFJS & Dataset...</p>
          </div>
        ) : meta && (
          <div className="flex flex-col lg:grid lg:grid-cols-3 gap-6 md:gap-8">
            <div className="lg:col-span-2 bg-white rounded-2xl md:rounded-3xl shadow-lg shadow-gray-200/50 p-5 md:p-8 border border-gray-100 order-last lg:order-none">
              
              {/* Wizard Progress Bar */}
              <div className="mb-8">
                <div className="flex justify-between mb-2 px-1">
                  <span className={`text-[10px] md:text-xs font-bold ${currentStep >= 1 ? 'text-red-600' : 'text-gray-400'}`}>1. Profil Dasar</span>
                  <span className={`text-[10px] md:text-xs font-bold ${currentStep >= 2 ? 'text-red-600' : 'text-gray-400'}`}>2. Masa Kerja</span>
                  <span className={`text-[10px] md:text-xs font-bold ${currentStep >= 3 ? 'text-red-600' : 'text-gray-400'}`}>3. Sentimen</span>
                </div>
                <div className="h-2 w-full bg-gray-100 rounded-full overflow-hidden flex">
                  <div className={`h-full bg-red-600 transition-all duration-500 ${currentStep === 1 ? 'w-1/3' : currentStep === 2 ? 'w-2/3' : 'w-full'}`}></div>
                </div>
              </div>

              <form onSubmit={handlePredict} className="space-y-5 md:space-y-6">
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 md:gap-5">
                  {currentFeatures.map(feature => {
                    const isCategorical = !!meta.categorical_mapping[feature];
                    const isEmotive = EMOTIVE_SCALES.includes(feature);
                    const labelName = feature.replace(/([A-Z])/g, ' $1').trim();
                    
                    return (
                      <div key={feature} className="space-y-1.5">
                        <label className="text-xs md:text-sm font-bold text-gray-800 truncate block" title={feature}>
                          {labelName}
                        </label>
                        
                        {isEmotive ? (
                           <select
                            name={feature}
                            value={formData[feature]}
                            onChange={handleChange}
                            className="w-full px-3 py-2.5 text-sm rounded-xl border border-gray-200 focus:ring-2 focus:ring-red-500 focus:border-red-500 bg-gray-50 outline-none transition-all cursor-pointer font-medium text-gray-700 shadow-sm"
                          >
                            <option value="" disabled>-- Pilih Opsi --</option>
                            <option value={1}>1 - Sangat Buruk 😡</option>
                            <option value={2}>2 - Buruk / Kurang 🙁</option>
                            <option value={3}>3 - Biasa Saja 😐</option>
                            <option value={4}>4 - Baik / Puas 🙂</option>
                            <option value={5}>5 - Sangat Baik 😍</option>
                          </select>
                        ) : feature === 'OverTime' ? (
                          <select
                            name={feature}
                            value={formData[feature]}
                            onChange={handleChange}
                            className="w-full px-3 py-2.5 text-sm rounded-xl border border-gray-200 focus:ring-2 focus:ring-red-500 focus:border-red-500 bg-gray-50 outline-none transition-all cursor-pointer font-medium text-gray-700 shadow-sm"
                          >
                            <option value="" disabled>-- Pilih Opsi --</option>
                            <option value="No">Tidak / Jarang</option>
                            <option value="Yes">Ya, Sering Lembur</option>
                          </select>
                        ) : isCategorical ? (
                          <select
                            name={feature}
                            value={formData[feature]}
                            onChange={handleChange}
                            className="w-full px-3 py-2.5 text-sm rounded-xl border border-gray-200 focus:ring-2 focus:ring-red-500 focus:border-red-500 bg-gray-50 outline-none transition-all cursor-pointer shadow-sm"
                          >
                            <option value="" disabled>-- Pilih Kategori --</option>
                            {Object.keys(meta.categorical_mapping[feature]).map(cat => (
                              <option key={cat} value={cat}>{cat}</option>
                            ))}
                          </select>
                        ) : (
                          <input
                            type="number"
                            name={feature}
                            value={formData[feature]}
                            onChange={handleChange}
                            step="any"
                            placeholder="Masukkan Angka..."
                            className="w-full px-3 py-2.5 text-sm rounded-xl border border-gray-200 focus:ring-2 focus:ring-red-500 focus:border-red-500 bg-gray-50 outline-none transition-all shadow-sm"
                          />
                        )}
                      </div>
                    );
                  })}
                </div>

                <div className="pt-6 mt-6 border-t flex flex-col-reverse sm:flex-row items-stretch sm:items-center justify-between gap-4">
                  {currentStep > 1 ? (
                    <button
                      type="button"
                      onClick={() => setCurrentStep(prev => (prev - 1) as 1|2|3)}
                      className="px-6 py-3.5 bg-gray-100 hover:bg-gray-200 text-gray-700 font-bold rounded-xl transition-all flex items-center justify-center gap-2 text-sm w-full sm:w-auto"
                    >
                      <ArrowLeft className="w-4 h-4" /> Kembali
                    </button>
                  ) : <div className="hidden sm:block"></div>}
                  
                  {currentStep < 3 ? (
                    <button
                      type="button"
                      onClick={() => setCurrentStep(prev => (prev + 1) as 1|2|3)}
                      disabled={!isStepValid(currentStep)}
                      className="px-8 py-3.5 bg-gray-800 hover:bg-gray-900 text-white font-bold rounded-xl transition-all shadow-lg flex items-center justify-center gap-2 text-sm w-full sm:w-auto disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      Selanjutnya <ArrowRight className="w-4 h-4" />
                    </button>
                  ) : (
                    <button
                      type="submit"
                      disabled={isPredicting || !isStepValid(3)}
                      className="bg-red-600 hover:bg-red-700 disabled:bg-gray-400 text-white font-bold py-3.5 px-8 rounded-xl transition-all shadow-lg shadow-red-200 hover:shadow-red-300 flex items-center justify-center gap-2 text-sm w-full sm:w-auto disabled:cursor-not-allowed disabled:shadow-none"
                    >
                      {isPredicting ? (
                        <div className="flex items-center gap-2">
                          <span>Menganalisis...</span>
                        </div>
                      ) : (
                        <>
                          <TrendingUp className="w-5 h-5" />
                          Prediksi Sekarang
                        </>
                      )}
                    </button>
                  )}
                </div>
              </form>
            </div>

            <div id="result-panel" className="lg:col-span-1 scroll-mt-6 md:scroll-mt-8 order-first lg:order-none">
              <div className="bg-white rounded-2xl md:rounded-3xl shadow-sm border border-gray-100 p-6 md:p-8 h-full flex flex-col justify-center text-center lg:sticky lg:top-8 min-h-[300px]">
                {prediction === null ? (
                  <div className="opacity-50 flex flex-col items-center transition-all">
                    <Activity className="w-12 h-12 md:w-16 md:h-16 text-gray-300 mb-3 md:mb-4" />
                    <p className="text-gray-500 text-sm font-medium">Isi form 3 tahap untuk melihat prediksi skor risiko.</p>
                  </div>
                ) : (
                  <div className="animate-in fade-in zoom-in duration-300 w-full text-center">
                    <h3 className="text-gray-500 font-bold mb-4 uppercase tracking-wider text-xs">Risk Score Prediction</h3>
                    
                    <div className="mb-6">
                      <div className="text-5xl font-bold text-gray-900 mb-1">
                        {Math.round(prediction * 100)}%
                      </div>
                    </div>

                    {(() => {
                      const prob = prediction * 100;
                      const isHighRisk = prob > 70;
                      const isMedRisk = prob >= 40 && prob <= 70;
                      
                      let badgeColors = 'bg-emerald-50 text-emerald-700 border-emerald-200';
                      let statusText = 'Kondisi Karyawan: Stabil (Risiko Rendah)';
                      let descText = 'Karyawan menunjukkan tingkat loyalitas dan kenyamanan yang baik di perusahaan saat ini.';
                      let saranHR = 'Pertahankan iklim kerja saat ini dan berikan apresiasi berkala agar motivasi tetap terjaga.';
                      let Icon = CheckCircle;
                      
                      if (isHighRisk) {
                        badgeColors = 'bg-red-50 text-red-600 border-red-200';
                        statusText = 'Kondisi Karyawan: Waspada Attrisi (Risiko Tinggi)';
                        descText = 'Karyawan berada dalam fase kritis dan memiliki indikasi kuat untuk meninggalkan perusahaan.';
                        saranHR = 'TINDAKAN SEGERA! Evaluasi beban kerja lembur (Overtime), tingkat kepuasan fasilitas, atau diskusikan kejelasan jalur karier bersama manajer terkait.';
                        Icon = AlertTriangle;
                      } else if (isMedRisk) {
                        badgeColors = 'bg-amber-50 text-amber-700 border-amber-200';
                        statusText = 'Kondisi Karyawan: Butuh Perhatian (Risiko Sedang)';
                        descText = 'Terdapat indikator kejenuhan atau ketidakpuasan minor yang mulai memengaruhi kenyamanan kerja.';
                        saranHR = 'Jalankan pendekatan personal atau diskusi santai (1-on-1) untuk mendengar kendala yang sedang dihadapi.';
                        Icon = Activity;
                      }

                      return (
                        <div className={`mt-2 p-5 rounded-2xl text-left border ${badgeColors}`}>
                          <div className="flex items-center gap-2 mb-2">
                            <Icon className="w-6 h-6 flex-shrink-0" />
                            <p className="font-bold text-sm md:text-base tracking-tight">{statusText}</p>
                          </div>
                          <p className="text-xs md:text-sm opacity-90 mb-4 font-medium leading-relaxed">
                            {descText}
                          </p>
                          
                          <div className="border-t border-gray-100 pt-4 mt-2">
                             <p className="text-[10px] md:text-xs font-black uppercase mb-2 flex items-center gap-1.5 opacity-80">
                               <Activity className="w-3.5 h-3.5"/> Saran HR
                             </p>
                             <p className="text-xs md:text-sm font-medium leading-relaxed opacity-90">
                               {saranHR}
                             </p>
                          </div>
                        </div>
                      );
                    })()}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
