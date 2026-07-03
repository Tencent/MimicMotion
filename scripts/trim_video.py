import argparse

import av


def trim_video(input_path, output_path, duration_sec):
    in_container = av.open(input_path)
    in_stream = in_container.streams.video[0]

    out_container = av.open(output_path, mode="w")
    out_stream = out_container.add_stream("libx264", rate=in_stream.average_rate)
    out_stream.width = in_stream.codec_context.width
    out_stream.height = in_stream.codec_context.height
    out_stream.pix_fmt = "yuv420p"

    for frame in in_container.decode(in_stream):
        if frame.time > duration_sec:
            break
        for packet in out_stream.encode(frame):
            out_container.mux(packet)

    for packet in out_stream.encode():
        out_container.mux(packet)

    out_container.close()
    in_container.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("--duration", type=float, default=4.0)
    args = parser.parse_args()

    trim_video(args.input, args.output, args.duration)
    print(f"Saved {args.duration}s clip to {args.output}")
