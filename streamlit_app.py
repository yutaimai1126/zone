import cv2
import streamlit as st
import pickle
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from utils import get_face_landmarks

st.set_page_config(page_title="Zone", page_icon='other_pic/icon.png')

def zone_estimation():
    emotions = ['ZONE', 'LAZY', 'NOMAL']
    
    with open('./model', 'rb') as f:
        model = pickle.load(f)

    webrtc_ctx = webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV)
    
    st.text('画像処理')
    if webrtc_ctx.video_receiver:
        frame_placeholder = st.empty()
        study_time_placeholder = st.empty()
        zone_time_placeholder = st.empty()
        lazy_time_placeholder = st.empty()
        nomal_time_placeholder = st.empty()
        focus_score_placeholder = st.empty()
        stage_placeholder = st.empty()

        # ストップボタンの作成とステート管理
        if 'stop_button_pressed' not in st.session_state:
            st.session_state.stop_button_pressed = False
        
        stop_button_pressed = st.button("Stop", key="stop_button")
        if stop_button_pressed:
            st.session_state.stop_button_pressed = True

        # 初期化
        if 'study_time' not in st.session_state:
            st.session_state.study_time = 0
        if 'zone_time' not in st.session_state:
            st.session_state.zone_time = 0
        if 'lazy_time' not in st.session_state:
            st.session_state.lazy_time = 0
        if 'nomal_time' not in st.session_state:
            st.session_state.nomal_time = 0
        if 'focus_score' not in st.session_state:
            st.session_state.focus_score = 100
        if 'stage' not in st.session_state:
            st.session_state.stage = None

        if not st.session_state.stop_button_pressed:
            frame = webrtc_ctx.video_receiver.recv()
            if frame is not None:
                # 顔ランドマークの取得
                face_landmarks = get_face_landmarks(frame, draw=False, static_image_mode=False)

                if face_landmarks:
                    output = model.predict([face_landmarks])
                    st.session_state.stage = emotions[int(output[0])]
                else:
                    st.session_state.stage = "normal"  # 顔が検出されない場合

                # 推定結果をフレームに表示
                cv2.putText(frame, st.session_state.stage, (10, frame.shape[0] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2, cv2.LINE_AA)

                # Streamlitに動画フレームを表示
                frame_placeholder.image(frame, channels="RGB")

                # タイム更新
                st.session_state.study_time += 1 / 30  # 仮のフレームレート
                if st.session_state.stage == 'LAZY':
                    st.session_state.lazy_time += 1 / 30
                elif st.session_state.stage == 'ZONE':
                    st.session_state.zone_time += 1 / 30
                else:
                    st.session_state.nomal_time += 1 / 30

                st.session_state.focus_score = (st.session_state.nomal_time * 0.8 + st.session_state.zone_time - st.session_state.lazy_time) * 100 / st.session_state.study_time

                study_time_placeholder.text(f'Study time: {str(round(st.session_state.study_time, 1))}')
                stage_placeholder.text(f'Stage: {st.session_state.stage}')
                nomal_time_placeholder.text(f'NOMAL time: {str(round(st.session_state.nomal_time, 1))}')
                lazy_time_placeholder.text(f'LAZY time: {str(round(st.session_state.lazy_time, 1))}')
                zone_time_placeholder.text(f'ZONE time: {str(round(st.session_state.zone_time, 1))}')
                focus_score_placeholder.text(f'Focus score: {str(round(st.session_state.focus_score, 0))}')
    
        if st.session_state.stop_button_pressed:
            st.text(f'勉強お疲れ様！あなたの勉強時間は{round(st.session_state.study_time, 0)}秒、集中度は{round(st.session_state.focus_score, 0)}%です！')


def main():
    st.title("集中力測定")
    zone_estimation()

if __name__ == '__main__':
    main()
