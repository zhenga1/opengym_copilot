import streamlit as st
import os
from run_episode import run_episode
from reward_wrapper import RewardShapingWrapper, run_episode_train, train_agent
import imageio
import time

# === HEADER === #
st.markdown("""
    <div style='background-color:#1e1e2f;padding:20px;border-radius:10px;margin-bottom:20px'>
        <h1 style='color:white;text-align:center;'>🤖 OpenGym Copilot</h1>
        <p style='color:#d3d3d3;text-align:center;font-size:16px;'>LLM-guided reward tuning & rollout visualization</p>
    </div>
""", unsafe_allow_html=True)

root_directory = "video_logs/"
def save_video(frames, video_path="rollout.mp4", fps=30):
    imageio.mimsave(os.path.join(root_directory, video_path), frames, fps=fps)

def render_frame_ui(suffix="", unique_name="iteration"):
    # unique_name is for the streamlit slider
    if "frames" in st.session_state and st.session_state["frames"]:
        st.markdown("""
            <div style='background-color:#f8f9fa;padding:20px;border-radius:10px;'>
                <h3 style='text-align:center;'>🎥 Episode Playback</h3>
            </div>
        """, unsafe_allow_html=True)
        # dividing the screen into col1 or col2
        #st.subheader(" 🎥 Episode Playback")
        #frames = st.session_state["frames"]
        #log = st.session_state["rewards_log"]
        #idx = st.session_state["frame_idx"]
        
        # if not st.session_state["play_mode"]:
        #     idx = st.slider(
        #         "Frame",
        #         0,
        #         len(st.session_state["frames"]) - 1,
        #         value=st.session_state["frame_idx"],
        #         key="frame_slider"
        #     )
        #     st.session_state["frame_idx"] = idx

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("▶️ Play", key=f"play_button_{suffix}"):
                st.session_state["play_mode"] = True
        with col2:
            if st.button("⏹️ Stop", key=f"stop_button_{suffix}"):
                st.session_state["play_mode"] = False
        with col3:
            if st.button("🔁 Reset", key=f"reset_button_{suffix}"):
                #st.session_state["frame_idx"] = 0 # this is for frame by frame analysis
                st.session_state["play_mode"] = False
        

        col1, col2 = st.columns([1, 1])
        
            # Show Frame
        #image_slot = st.empty()
        with col1:
            if st.button("⬅️ Prev", key=f"prev_button_{suffix}") and  st.session_state["frame_idx"] > 0:
                st.session_state["frame_idx"] -= 1
                #frame = st.session_state["frames"][st.session_state["frame_idx"]]
                #image_slot.image(frame)
                #st.rerun()
        with col2:
            if st.button("➡️ Next", key=f"next_button_{suffix}") and  st.session_state["frame_idx"] < len(st.session_state["frames"]) - 1:
                st.session_state["frame_idx"] += 1
                #frame = st.session_state["frames"][st.session_state["frame_idx"]]
                #image_slot.image(frame)
                #st.rerun()
        playback_fps = st.slider("Playback FPS", min_value=30, max_value=120, value=45, key=f"playback_fps_{suffix}")
        
        if st.button("Save Episodes", key=f"save_episodes_{suffix}"):
            st.session_state["show_save_inputs"] = True
        
        ## Playback logic
        image_placeholder = st.empty()
        if st.session_state["play_mode"]:
            # if st.session_state["frame_idx"] < len(st.session_state["frames"]) - 1:
            #     st.session_state["frame_idx"] += 1
            #     time.sleep(frame_delay) # each frame takes 1/30 seconds
            #     st.rerun()
            # else:
            #     st.session_state["play_mode"] = False # stop at the end
            for idx, frame in enumerate(st.session_state["frames"]):
                image_placeholder.image(frame, channels="RGB", use_container_width=True)
                time.sleep(1.0 / playback_fps)

        if st.session_state["show_save_inputs"]:
            video_name = st.text_input("Enter filename (no extension):", value="bipedal_rollout", key=f"bipedal_rollout_{suffix}")
            file_format = st.selectbox("File format", ["mp4", "gif"], key=f"file_format_select_{suffix}")
            filename = f"{video_name}.{file_format}"
            if st.button("Confirm and Save", key=f"confirm_save_{suffix}"):
                save_video(st.session_state["frames"], video_path=filename, fps=playback_fps)
                st.success(f"Saved as {filename}")

                # two second rest
                time.sleep(2)
                st.session_state["show_save_inputs"] = False
                # close windoow
                st.rerun()

        frame_delay = 1.0 / playback_fps
        
        # Place the image_slot **after** the buttons
        frame = st.session_state["frames"][st.session_state["frame_idx"]]
        image_slot = st.empty()
        image_slot.image(frame, channels="RGB", use_container_width=True)
        #st.image(frame, channels="RGB", use_container_width=True)
        

        #Reward Info
        # add the info of the reward into the st
        if st.session_state["frame_idx"] < len(st.session_state["all_reward_logs"]):
            #import pdb
            #pdb.set_trace()
            idx = st.session_state["frame_idx"]
            #reward, shaped, info = st.session_state["rewards_log"][st.session_state["frame_idx"]]
            reward = st.session_state["total_reward"]
            shaped = st.session_state["total_shaped_reward"]
            info = st.session_state["all_reward_logs"][idx]
            st.markdown(f"**Raw Reward:** {reward:.3f} | **Shaped:** {shaped:.3f}")
            #st.json(info) no need to show the info
        
        
        

        st.subheader("Scrollable Frame Viewer")
        # Prev / Next buttons
        
        idx = st.slider("Frame Index", 0, len(st.session_state["frames"]) - 1, 0, key=unique_name)
        st.image(st.session_state["frames"][idx],
                    caption=f"Frame {idx + 1} / {len(st.session_state['frames'])}", 
                    channels="RGB", 
                    use_column_width=True)
        

        # for frame in frames:
        #     st.image(frame, channels="RGB", use_column_width=True)
        #     time.sleep(1/30) # sleep for 1/30th of a second


#st.set_page_config(layout="wide")
#st.title(" 🤖s OpenGym Copilot: Reward Tuner")

#all_envs = ['CartPole-v0', 'CartPole-v1', 'MountainCar-v0', 'MountainCarContinuous-v0', 'Pendulum-v1', 'Acrobot-v1', 'phys2d/CartPole-v0', 'phys2d/CartPole-v1', 'phys2d/Pendulum-v0', 'LunarLander-v3', 'LunarLanderContinuous-v3', 'BipedalWalker-v3', 'BipedalWalkerHardcore-v3', 'CarRacing-v3', 'Blackjack-v1', 'FrozenLake-v1', 'FrozenLake8x8-v1', 'CliffWalking-v0', 'Taxi-v3', 'tabular/Blackjack-v0', 'tabular/CliffWalking-v0', 'Reacher-v2', 'Reacher-v4', 'Reacher-v5', 'Pusher-v2', 'Pusher-v4', 'Pusher-v5', 'InvertedPendulum-v2', 'InvertedPendulum-v4', 'InvertedPendulum-v5', 'InvertedDoublePendulum-v2', 'InvertedDoublePendulum-v4', 'InvertedDoublePendulum-v5', 'HalfCheetah-v2', 'HalfCheetah-v3', 'HalfCheetah-v4', 'HalfCheetah-v5', 'Hopper-v2', 'Hopper-v3', 'Hopper-v4', 'Hopper-v5', 'Swimmer-v2', 'Swimmer-v3', 'Swimmer-v4', 'Swimmer-v5', 'Walker2d-v2', 'Walker2d-v3', 'Walker2d-v4', 'Walker2d-v5', 'Ant-v2', 'Ant-v3', 'Ant-v4', 'Ant-v5', 'Humanoid-v2', 'Humanoid-v3', 'Humanoid-v4', 'Humanoid-v5', 'HumanoidStandup-v2', 'HumanoidStandup-v4', 'HumanoidStandup-v5', 'GymV21Environment-v0', 'GymV26Environment-v0']
env_name = st.selectbox("Environment", ["BipedalWalker-v3", "LunarLanderContinuous-v3", "Ant-v4", "HalfCheetah-v4", "Walker2d-v4", "Humanoid-v4", "Pendulum-v1"])
episodes = st.slider("Number of Episodes:", 1, 10, 3)

st.markdown("### 🎯 Reward Weights")
forward_velocity = st.slider("Forward Velocity", -2.0, 2.0, 1.0)
energy_penalty = st.slider("Energy Penalty", -2.0, 2.0, -0.1)
alive_bonus = st.slider("Alive Bonus", 0.0, 2.0, 1.0)

# === SESSION STATE INIT === #
for key, default in {
    "show_save_inputs": False,
    "frames": [],
    "all_reward_logs": [],
    "frame_idx": 0,
    "play_mode": False,
    "train_mode_epoch_size_defined": False,
    "timesteps_str": 1000,
    "total_shaped_reward": 0,
    "total_reward": 0
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

stopped = False
#playback_fps = st.slider("🎞️ Playback FPS", min_value=10, max_value=120, value=45)

largest_playback_fps = 120

if st.button("Run Episodes") and not stopped:
    weights = {
        "forward_velocity": forward_velocity,
        "energy_penalty": energy_penalty,
        "alive_bonus": alive_bonus
    }
    with st.spinner("Running environments..."):
        total_shaped_reward, total_reward, all_reward_logs, frames = run_episode(env_name, weights, episodes, render_mode="rgb_array")
        st.success(f"Total Shaped Reward: {total_shaped_reward:.2f}")
        st.session_state["frames"] = frames # store safely 
        st.session_state["all_reward_logs"] = all_reward_logs
        st.session_state["total_shaped_reward"] = total_shaped_reward
        st.session_state["total_reward"] = total_reward
        st.session_state["frame_idx"] = 0
        st.session_state["play_mode"] = False
        #st.success(f"✅ Total Shaped Reward: {:.2f}")
        st.success(f"✅ Total Reward: {total_reward:.2f}")
        video_path = save_video(frames, "latest_rollout.mp4", fps=largest_playback_fps)

# render frame UI for the first appearance
render_frame_ui("1", "first")
    #import pdb
    #pdb.set_trace()
    # Animate the frames

if st.button("🧠 Train Agent with Current Reward Weights"):
    st.text_input("Enter timesteps (integer):", value="1000", key="timesteps_str")
    st.session_state["train_mode_epoch_size_defined"] = True

if st.session_state["train_mode_epoch_size_defined"]:
    # Convert to int safely
    timesteps_str = st.session_state["timesteps_str"]
    if st.button("Confirm and Save"):
        try:
            total_timesteps = int(timesteps_str)
        except ValueError:
            st.error("Please enter a valid integer for timesteps.")
            total_timesteps = 1000  # fallback default

        with st.spinner("Training PPO agent..."):
            st.write(f"Training started! Timesteps is {total_timesteps}")
            model = train_agent(env_name, {
                "forward_velocity": forward_velocity,
                "energy_penalty": energy_penalty,
                "alive_bonus": alive_bonus
            }, total_timesteps=total_timesteps)

            st.success("Training complete ✅")

            # Run new episode with trained model
            raw_reward, log, frames = run_episode_train(
                env_name,
                {
                    "forward_velocity": forward_velocity,
                    "energy_penalty": energy_penalty,
                    "alive_bonus": alive_bonus
                },
                episodes=1,
                model=model
            )
            st.session_state["frames"] = frames
            st.session_state["all_reward_logs"] = log
            st.session_state["frame_idx"] = 0
            #st.session_state["total_shaped_reward"] = shaped_reward
            st.session_state["total_reward"] = raw_reward

        # two second rest
        time.sleep(2)
        st.session_state["train_mode_epoch_size_defined"] = False

# render frame UI for the second appearance
render_frame_ui("2", "second")

