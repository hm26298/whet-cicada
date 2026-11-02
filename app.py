from pathlib import Path
import io
import hashlib
import zipfile

import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parent
MODEL_NAME = "걸생 베타 매미 종 동정 모델.keras"
CLASSES_NAME = "classes.npy"
CONFIG_NAME = "feature_config.npz"
CLASSES = ["말매미", "참매미", "쓰름매미", "유지매미", "털매미", "애매미", "배경소리"]
SAMPLE_RATE = 44100
N_MFCC = 40
# 지나치게 긴 파일이 서버 메모리를 소모하지 않도록 제한합니다.
MAX_DURATION = 120
MAX_FILE_BYTES = 30 * 1024 * 1024


def extract_features(data, sample_rate, n_mfcc):
    if len(data) > MAX_FILE_BYTES:
        raise ValueError("음원은 파일당 30MB 이하로 올려 주세요.")
    # 제한보다 조금 더 읽어 너무 긴 음원을 감지합니다. 묵시적으로 자르지 않습니다.
    audio, sr = librosa.load(
        io.BytesIO(data), sr=sample_rate, mono=True, duration=MAX_DURATION + 1
    )
    if audio.size == 0:
        raise ValueError("비어 있는 음원입니다.")
    if not np.all(np.isfinite(audio)):
        raise ValueError("오디오에 잘못된 값이 있습니다.")
    duration = len(audio) / sr
    if duration > MAX_DURATION:
        raise ValueError(f"{MAX_DURATION}초 이하의 음원을 사용해 주세요.")
    if np.max(np.abs(audio)) == 0:
        raise ValueError("완전히 무음인 파일은 사용할 수 없습니다.")
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    feature = np.mean(mfcc, axis=1).astype(np.float32)
    if not np.all(np.isfinite(feature)):
        raise ValueError("특징 계산 결과가 올바르지 않습니다.")
    return feature, duration


def validate_bundle(model, categories, sample_rate, n_mfcc):
    if categories.ndim != 1 or len(categories) < 2:
        raise ValueError("카테고리 목록이 올바르지 않습니다.")
    if categories.dtype.kind not in "US":
        raise ValueError("카테고리 이름은 문자열이어야 합니다.")
    if len(set(categories.tolist())) != len(categories):
        raise ValueError("카테고리 이름이 중복됩니다.")
    if not 1000 <= sample_rate <= 192000 or not 1 <= n_mfcc <= 256:
        raise ValueError("특징 추출 설정이 허용 범위를 벗어났습니다.")
    if isinstance(model.input_shape, list) or isinstance(model.output_shape, list):
        raise ValueError("단일 입력·출력 모델만 지원합니다.")
    if len(model.input_shape) != 2 or model.input_shape[-1] != n_mfcc:
        raise ValueError("모델과 MFCC 설정이 일치하지 않습니다.")
    if len(model.output_shape) != 2 or model.output_shape[-1] != len(categories):
        raise ValueError("모델과 카테고리 목록이 일치하지 않습니다.")


# 공유 캐시에 모델을 보관하지 않습니다. 학습 결과는 방문자의 세션에만 둡니다.
def load_repository_bundle():
    paths = [BASE_DIR / name for name in (MODEL_NAME, CLASSES_NAME, CONFIG_NAME)]
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("저장소에 없는 파일: " + ", ".join(missing))
    model = tf.keras.models.load_model(str(paths[0]), compile=False, safe_mode=True)
    categories = np.load(str(paths[1]), allow_pickle=False)
    with np.load(str(paths[2]), allow_pickle=False) as config:
        sr = int(config["sample_rate"].item())
        n_mfcc = int(config["n_mfcc"].item())
    validate_bundle(model, categories, sr, n_mfcc)
    return model, categories, sr, n_mfcc


def package_bundle(model, categories, sample_rate, n_mfcc):
    # Keras는 .keras 파일 경로가 필요하므로 임시 폴더를 사용합니다.
    import tempfile
    with tempfile.TemporaryDirectory() as folder:
        model_path = Path(folder) / MODEL_NAME
        model.save(str(model_path))
        class_buffer = io.BytesIO()
        np.save(class_buffer, np.asarray(categories, dtype=str), allow_pickle=False)
        config_buffer = io.BytesIO()
        np.savez(config_buffer, sample_rate=sample_rate, n_mfcc=n_mfcc)
        result = io.BytesIO()
        with zipfile.ZipFile(result, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(model_path, MODEL_NAME)
            archive.writestr(CLASSES_NAME, class_buffer.getvalue())
            archive.writestr(CONFIG_NAME, config_buffer.getvalue())
        return result.getvalue()


def show_classification():
    st.subheader("녹음 파일 분류")
    source = st.radio(
        "사용할 모델", ["GitHub에 저장된 모델", "이번 세션에서 학습한 모델"],
        horizontal=True,
    )
    uploaded = st.file_uploader(
        "분류할 MP3 또는 WAV 파일", type=["mp3", "wav"], key="classify_audio"
    )
    st.caption("파일당 최대 30MB, 120초. 학습 때와 동일하게 파일 전체의 MFCC 평균을 사용합니다.")
    if uploaded is not None:
        st.audio(uploaded.getvalue())
    if st.button("매미 종류 분석", type="primary", disabled=uploaded is None):
        try:
            with st.spinner("모델을 준비하고 음원을 분석하고 있습니다…"):
                if source == "이번 세션에서 학습한 모델":
                    bundle = st.session_state.get("trained_bundle")
                    if bundle is None:
                        raise ValueError("먼저 '음원 추가·학습' 탭에서 학습해 주세요.")
                else:
                    if "repository_bundle" not in st.session_state:
                        st.session_state.repository_bundle = load_repository_bundle()
                    bundle = st.session_state.repository_bundle
                model, categories, sr, n_mfcc = bundle
                feature, duration = extract_features(uploaded.getvalue(), sr, n_mfcc)
                scores = np.asarray(
                    model.predict(feature.reshape(1, -1), verbose=0)[0], dtype=np.float64
                )
                if scores.shape != (len(categories),) or not np.all(np.isfinite(scores)):
                    raise ValueError("모델 예측 결과가 올바르지 않습니다.")
                if np.any(scores < 0) or np.any(scores > 1) or not np.isclose(scores.sum(), 1, atol=1e-3):
                    raise ValueError("모델 출력이 softmax 분류 점수 형식이 아닙니다.")
                order = np.argsort(scores)[::-1]
                best = int(order[0])
            st.success(f"예측 결과: {categories[best]}")
            st.metric("최상위 모델 예측 점수", f"{scores[best] * 100:.1f}%")
            st.dataframe(
                [{"순위": rank, "종류": str(categories[i]),
                  "모델 예측 점수 (%)": round(float(scores[i]) * 100, 2)}
                 for rank, i in enumerate(order, start=1)],
                hide_index=True, use_container_width=True,
            )
            st.caption(f"녹음 길이: {duration:.1f}초 · 분류 가능한 종류: {', '.join(map(str, categories))}")
        except Exception as error:
            st.error(f"분석 실패: {error}")
    st.info(
        "모델 점수는 실제 정확도나 정답 확률을 보장하지 않습니다. "
        "학습하지 않은 소리도 등록된 종류 중 하나로 분류될 수 있습니다. "
        "여러 종이 함께 울어도 이 모델은 파일 전체에 대해 하나의 대표 결과를 냅니다."
    )


def show_training():
    st.subheader("음원 추가·학습")
    st.warning(
        "추가한 음원과 학습 결과는 현재 방문자의 세션에서만 사용됩니다. "
        "접속이 끊기거나 서버가 재시작되면 사라질 수 있습니다. "
        "학습 후 결과 ZIP을 반드시 내려받으세요. GitHub에는 자동 저장되지 않습니다."
    )
    st.caption("공유 서버의 자원을 사용하므로 소규모 실험용입니다. 본격적인 학습은 개인 컴퓨터에서 권장합니다.")
    if "training_samples" not in st.session_state:
        st.session_state.training_samples = {}
    samples = st.session_state.training_samples
    category = st.selectbox("등록할 카테고리", CLASSES)
    uploads = st.file_uploader(
        "같은 종류의 음원을 선택하세요", type=["mp3", "wav"],
        accept_multiple_files=True, key="training_uploads",
    )
    st.caption("파일당 최대 30MB, 120초. 한 세션에 최대 200개까지 등록할 수 있습니다.")
    if st.button("선택한 음원 등록", disabled=not uploads):
        added = skipped = failed = 0
        progress = st.progress(0)
        for idx, upload in enumerate(uploads):
            try:
                data = upload.getvalue()
                digest = hashlib.sha256(data).hexdigest()
                if digest in samples:
                    skipped += 1
                    if samples[digest]["category"] != category:
                        st.warning(f"{upload.name}: 이미 다른 카테고리로 등록되어 제외했습니다.")
                else:
                    if len(samples) >= 200:
                        raise ValueError("한 세션에는 최대 200개 음원만 등록할 수 있습니다.")
                    feature, _ = extract_features(data, SAMPLE_RATE, N_MFCC)
                    samples[digest] = {"feature": feature, "category": category, "name": upload.name}
                    added += 1
            except Exception as error:
                failed += 1
                st.error(f"{upload.name}: {error}")
            progress.progress((idx + 1) / len(uploads))
        progress.empty()
        st.success(f"등록 {added}개 / 중복 제외 {skipped}개 / 실패 {failed}개")
    st.dataframe(
        [{"종류": c, "등록 음원 수": sum(s["category"] == c for s in samples.values())}
         for c in CLASSES], hide_index=True, use_container_width=True,
    )
    st.write(f"전체 등록 음원: **{len(samples)}개**")
    if st.checkbox("등록한 학습 데이터를 모두 비우겠습니다"):
        if st.button("학습 데이터 비우기"):
            st.session_state.training_samples = {}
            st.rerun()
    epochs = st.slider("학습 반복 횟수", min_value=5, max_value=100, value=50, step=5)
    st.caption("기존 모델에 이어서 학습하지 않고, 현재 등록 음원으로 새 모델을 만듭니다.")
    if st.button("등록 음원으로 새 모델 학습", type="primary"):
        labels = [s["category"] for s in samples.values()]
        if len(set(labels)) < 2:
            st.error("최소 2개 카테고리에 음원을 등록해 주세요.")
            return
        if min(labels.count(c) for c in set(labels)) < 5:
            st.warning("일부 카테고리의 음원이 5개 미만입니다. 실행은 가능하지만 성능을 신뢰하기 어렵습니다.")
        try:
            tf.keras.utils.set_random_seed(42)
            X = np.asarray([s["feature"] for s in samples.values()], dtype=np.float32)
            encoder = LabelEncoder()
            y = encoder.fit_transform(labels)
            normalizer = tf.keras.layers.Normalization(axis=-1)
            normalizer.adapt(X)
            model = tf.keras.Sequential([
                tf.keras.Input(shape=(N_MFCC,)), normalizer,
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(len(encoder.classes_), activation="softmax"),
            ])
            model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            progress = st.progress(0)
            status = st.empty()

            class ProgressCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    logs = logs or {}
                    progress.progress((epoch + 1) / epochs)
                    status.write(
                        f"학습 {epoch + 1}/{epochs} · "
                        f"학습 정확도 {logs.get('accuracy', 0):.1%} · "
                        f"손실 {logs.get('loss', float('nan')):.4f}"
                    )

            with st.spinner("학습 중입니다. 이 페이지를 유지해 주세요…"):
                history = model.fit(
                    X, y, epochs=epochs, batch_size=4, shuffle=True, verbose=0,
                    callbacks=[tf.keras.callbacks.TerminateOnNaN(), ProgressCallback()],
                )
                losses = history.history.get("loss", [])
                if not losses or not np.all(np.isfinite(losses)) or not all(
                    np.all(np.isfinite(w)) for w in model.get_weights()
                ):
                    raise ValueError("학습 결과에 잘못된 수치가 있어 저장하지 않습니다.")
                categories = np.asarray(encoder.classes_, dtype=str)
                validate_bundle(model, categories, SAMPLE_RATE, N_MFCC)
                archive = package_bundle(model, categories, SAMPLE_RATE, N_MFCC)
            st.session_state.trained_bundle = (model, categories, SAMPLE_RATE, N_MFCC)
            st.session_state.model_zip = archive
            st.success("학습 완료! 분류 탭에서 '이번 세션에서 학습한 모델'을 선택해 사용할 수 있습니다.")
            st.write("학습한 종류: " + ", ".join(categories))
            st.info("표시된 정확도는 학습 데이터의 정확도입니다. 별도의 테스트 녹음으로 실제 성능을 확인하세요.")
        except Exception as error:
            st.error(f"학습 또는 결과 저장 실패: {error}")
    if "model_zip" in st.session_state:
        st.download_button(
            "마지막으로 완료한 학습 결과 ZIP 내려받기", st.session_state.model_zip,
            file_name="cicada_model.zip", mime="application/zip",
        )
        st.caption("ZIP을 풀어 나온 파일 3개를 GitHub의 app.py와 같은 폴더에 올리세요. ZIP 자체만 올리면 읽을 수 없습니다.")


st.set_page_config(page_title="매미 종 동정 프로그램", page_icon="🌿", layout="centered")
st.title("🌿 매미 종 동정 프로그램")
st.write("매미 녹음 파일을 분류하거나, 직접 모은 음원으로 간단한 모델을 학습할 수 있습니다.")
classify_tab, train_tab = st.tabs(["녹음 파일 분류", "음원 추가·학습"])
with classify_tab:
    show_classification()
with train_tab:
    show_training()
