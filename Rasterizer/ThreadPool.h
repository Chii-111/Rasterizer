#pragma once
// Worker threads wait on a condition variable for tasks
// Main thread pushes tasks and calls waitFinished() before present()
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>


class ThreadPool {
public:
  // Get singleton instance
  static ThreadPool &getInstance() {
    static ThreadPool instance;
    return instance;
  }

  // Initialize thread pool 
  void start(int numThreads = 0) {
    if (numThreads == 0)
      numThreads = std::thread::hardware_concurrency();
    // Leave one core for main thread to prevent stutter
    if (numThreads > 1)
      numThreads -= 1;

    stop = false;
    active_tasks = 0;

    for (int i = 0; i < numThreads; ++i) {
      workers.emplace_back([this] {
        while (true) {
          std::function<void()> task;
          {
            std::unique_lock<std::mutex> lock(queue_mutex);
            // Wait for task or stop signal
            condition.wait(lock, [this] { return stop || !tasks.empty(); });

            if (stop && tasks.empty())
              return;

            task = std::move(tasks.front());
            tasks.pop();
            active_tasks++;
          }

          task(); // Execute task

          {
            std::unique_lock<std::mutex> lock(queue_mutex);
            active_tasks--;
          }
          finished_condition.notify_all();
        }
      });
    }
  }

  // Add a task to the queue
  void enqueue(std::function<void()> task) {
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      tasks.emplace(std::move(task));
    }
    condition.notify_one();
  }

  // Wait for all tasks to complete 
  void waitFinished() {
    std::unique_lock<std::mutex> lock(queue_mutex);
    finished_condition.wait(lock, [this] { return tasks.empty() && (active_tasks == 0); });
  }

  // Get number of worker threads
  int getNumWorkers() const { return static_cast<int>(workers.size()); }

  // Destructor
  ~ThreadPool() {
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      stop = true;
    }
    condition.notify_all();
    for (std::thread &worker : workers) {
      if (worker.joinable())
        worker.join();
    }
  }

private:
  ThreadPool() {} // Private constructor for singleton

  std::vector<std::thread> workers;
  std::queue<std::function<void()>> tasks;

  std::mutex queue_mutex;
  std::condition_variable condition;
  std::condition_variable finished_condition;

  bool stop = false;
  int active_tasks = 0;
};
