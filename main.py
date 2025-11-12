from environs import Env

env = Env()
env.read_env()


def main():
    print("Hello from langchain-course!")


if __name__ == "__main__":
    main()
