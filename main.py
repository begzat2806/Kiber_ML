"""
main.py
─────────────────────────────────────────────────────────
РОЛЬ В ПРОЕКТЕ:
    Точка входа. Соединяет все модули вместе.
    Это ТВОЯ задача — здесь ты выступаешь как ML engineer,
    который собирает систему из готовых компонентов.

ЗАВИСИМОСТИ (все модули проекта):
    config.py   → get_config, validate_config, PATHS, TRAIN_CFG
    dataset.py  → get_or_generate_graph_data, create_dataloaders
    model.py    → build_model
    train.py    → Trainer, build_optimizer, build_scheduler, build_criterion
                  evaluate_on_test
    inference.py→ ThreatAnalyzer
    utils.py    → setup_logging, set_seed, get_device,
                  plot_training_curves, plot_confusion_matrix,
                  plot_semantic_heatmap

ПОРЯДОК СБОРКИ (твоя задача заполнить TODO-блоки):
    1. setup_logging + set_seed
    2. validate_config + логировать конфиг
    3. get_device
    4. get_or_generate_graph_data  ← данные + граф
    5. create_dataloaders          ← DataLoader'ы
    6. build_model + .to(device)
    7. build_optimizer(model)
    8. build_scheduler(optimizer)
    9. build_criterion(graph_data, device)
   10. Trainer(model, optimizer, scheduler, ...) + .fit()
   11. evaluate_on_test + plot_confusion_matrix
   12. ThreatAnalyzer + примеры инференса
   13. plot_training_curves + plot_semantic_heatmap

ЧТО УЖЕ СДЕЛАНО:
    - Весь скелет с правильным порядком вызовов
    - run_training() / run_inference() / run_full_pipeline()
    - CLI (argparse) для управления режимами
    - Обработка исключений и финальный report

ЧТО ДЕЛАЕШЬ ТЫ:
    - Заполняешь TODO-блоки (отмечены ★)
    - Подключаешь build_optimizer / build_scheduler
      (которые реализовал в train.py)
    - Запускаешь и отлаживаешь
─────────────────────────────────────────────────────────
"""

import argparse
import json
import logging
import os
import sys

import torch

# ── Импорты модулей проекта ───────────────────────────────────
# Каждый import — отдельный независимый модуль
from config import (
    get_config,
    validate_config,
    PATHS,
    TRAIN_CFG,
    DATA_CFG,
    INFERENCE_CFG,
)
from dataset import get_or_generate_graph_data, create_dataloaders
from model import build_model
from train import (
    Trainer,
    build_optimizer,    # ★ ты реализовал в train.py
    build_scheduler,    # ★ ты реализовал в train.py
    build_criterion,
    evaluate_on_test,
)
from inference import ThreatAnalyzer
from utils import (
    setup_logging,
    set_seed,
    get_device,
    plot_training_curves,
    plot_confusion_matrix,
    plot_semantic_heatmap,
    format_metrics,
    count_model_params,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# РЕЖИМ 1: ПОЛНЫЙ ПАЙПЛАЙН (обучение + тест + инференс)
# ══════════════════════════════════════════════════════════════

def run_full_pipeline(args: argparse.Namespace) -> None:
    """
    Последовательно запускает: подготовку данных → обучение →
    тестирование → инференс → визуализацию.

    Вызов:
        python main.py --mode full
        python main.py --mode full --regenerate_data
    """

    # ── Шаг 1: Инфраструктура ─────────────────────────────────
    logger.info("=" * 60)
    logger.info("CyberThreat Intelligence AI — Full Pipeline")
    logger.info("=" * 60)

    set_seed(TRAIN_CFG.seed)
    device = get_device()

    # Создаём папки если не существуют
    for path_key in ["data_dir", "checkpoints_dir", "logs_dir", "plots_dir"]:
        os.makedirs(PATHS[path_key], exist_ok=True)

    # ── Шаг 2: Логируем конфиг эксперимента ───────────────────
    cfg_dict = get_config()
    logger.info("Config:\n%s", json.dumps(
        {k: v for k, v in cfg_dict.items() if k != "paths"},
        indent=2, default=str
    ))

    # ── Шаг 3: Данные + Knowledge Graph ───────────────────────
    logger.info("─── Step 1/5: Loading data & Knowledge Graph ───")
    graph_data = get_or_generate_graph_data(force=args.regenerate_data)

    logger.info(
        "Graph: %d nodes | %d edges | %d incident samples",
        graph_data["n_classes"],
        graph_data["edge_index"].shape[1],
        len(graph_data["incident_X"]),
    )

    # Визуализируем семантическую матрицу сразу
    plot_semantic_heatmap(
        graph_data["sim_matrix"],
        graph_data["class_names"],
    )

    # ── Шаг 4: DataLoaders ────────────────────────────────────
    logger.info("─── Step 2/5: Creating DataLoaders ───")
    train_loader, val_loader, test_loader = create_dataloaders(
        graph_data   = graph_data,
        batch_size   = TRAIN_CFG.batch_size,
        augment_train= True,
    )

    # ── Шаг 5: Модель ─────────────────────────────────────────
    logger.info("─── Step 3/5: Building model ───")

    # ★ TODO: Создай модель и перенеси на device
    # model = build_model()
    # model = model.to(device)
    #
    # Проверь параметры:
    # params = count_model_params(model)
    # logger.info("Parameters: %s", params)
    raise NotImplementedError(
        "★ [Шаг 5] Создай модель:\n"
        "    model = build_model()\n"
        "    model = model.to(device)\n"
        "Затем удали этот raise и продолжи."
    )

    # ── Шаг 6: Optimizer ──────────────────────────────────────
    # ★ TODO: Подключи build_optimizer из train.py
    #
    # После того как реализовал build_optimizer() в train.py:
    # optimizer = build_optimizer(model)
    #
    # Раскомментируй и замени заглушку:
    optimizer = None   # ← замени на build_optimizer(model)

    # ── Шаг 7: Scheduler ──────────────────────────────────────
    # ★ TODO: Подключи build_scheduler из train.py
    #
    # После реализации build_scheduler() в train.py:
    # scheduler = build_scheduler(optimizer)
    #
    scheduler = None   # ← замени на build_scheduler(optimizer)

    # ── Шаг 8: Loss function ──────────────────────────────────
    criterion = build_criterion(graph_data, device)

    # ── Шаг 9: Trainer ────────────────────────────────────────
    logger.info("─── Step 4/5: Training ───")

    # ★ TODO: Создай Trainer и запусти обучение
    #
    # trainer = Trainer(
    #     model        = model,
    #     optimizer    = optimizer,
    #     scheduler    = scheduler,
    #     criterion    = criterion,
    #     train_loader = train_loader,
    #     val_loader   = val_loader,
    #     graph_data   = graph_data,
    #     device       = device,
    # )
    # history = trainer.fit()
    #
    # Если хочешь продолжить с чекпоинта:
    # from utils import CheckpointManager
    # ckpt_mgr    = CheckpointManager()
    # start_epoch, _ = ckpt_mgr.load(model, optimizer, scheduler)
    # history = trainer.fit(start_epoch=start_epoch)

    # ── Шаг 10: Визуализация кривых обучения ──────────────────
    # ★ TODO: Раскомментируй после запуска обучения
    # plot_training_curves()

    # ── Шаг 11: Тест на hold-out выборке ──────────────────────
    logger.info("─── Step 5/5: Testing ───")

    # ★ TODO: Раскомментируй после реализации обучения
    #
    # # Загружаем лучшую модель перед тестом
    # from utils import CheckpointManager
    # ckpt_mgr = CheckpointManager()
    # ckpt_mgr.load(model, path=PATHS["best_model"])
    #
    # test_metrics, cm = evaluate_on_test(
    #     model         = model,
    #     test_loader   = test_loader,
    #     criterion     = criterion,
    #     node_features = graph_data["node_features"].to(device),
    #     edge_index    = graph_data["edge_index"].to(device),
    #     device        = device,
    #     n_classes     = DATA_CFG.num_classes,
    # )
    # plot_confusion_matrix(cm, graph_data["class_names"])
    # logger.info("Final test: %s", format_metrics(test_metrics, "Test"))

    # ── Шаг 12: Примеры инференса ─────────────────────────────
    run_inference_examples(graph_data, device)

    logger.info("=" * 60)
    logger.info("Pipeline complete.")
    logger.info("=" * 60)


# ══════════════════════════════════════════════════════════════
# РЕЖИМ 2: ТОЛЬКО ОБУЧЕНИЕ
# ══════════════════════════════════════════════════════════════

def run_training(args: argparse.Namespace) -> None:
    set_seed(TRAIN_CFG.seed)
    device     = get_device()
    graph_data = get_or_generate_graph_data()

    train_loader, val_loader, test_loader = create_dataloaders(graph_data)
    
    model     = build_model().to(device)
    params    = count_model_params(model)
    logger.info("Parameters: total=%s trainable=%s",
                f"{params['total']:,}", f"{params['trainable']:,}")

    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer)
    criterion = build_criterion(graph_data, device)

    start_epoch = 0
    if args.resume:
        from utils import CheckpointManager
        start_epoch, _ = CheckpointManager().load(
            model, optimizer, scheduler
        )

    trainer = Trainer(
        model        = model,
        optimizer    = optimizer,
        scheduler    = scheduler,
        criterion    = criterion,
        train_loader = train_loader,
        val_loader   = val_loader,
        graph_data   = graph_data,
        device       = device,
    )
    trainer.fit(start_epoch=start_epoch)
    from utils import CheckpointManager
    ckpt_mgr = CheckpointManager()
    ckpt_mgr.load(model, path=PATHS["best_model"])

    test_metrics, cm = evaluate_on_test(
        model         = model,
        test_loader   = test_loader,          # нужно вернуть из create_dataloaders
        criterion     = criterion,
        node_features = graph_data["node_features"].to(device),
        edge_index    = graph_data["edge_index"].to(device),
        device        = device,
        n_classes     = DATA_CFG.num_classes,
    )
    plot_confusion_matrix(cm, graph_data["class_names"])
    # ── Инференс на примерах из тестовой выборки ──
    logger.info("--- Inference examples ---")
    analyzer = ThreatAnalyzer.from_checkpoint(
        checkpoint_path = PATHS["best_model"],
        graph_data      = graph_data,
        device          = device,
    )

    # Берём по 1 примеру каждого класса из тестовых данных
    test_X = graph_data["incident_X"]
    test_y = graph_data["incident_y"]

    shown = set()
    for i in range(len(test_X)):
        cls = test_y[i].item()
        if cls not in shown:
            result = analyzer.analyze(test_X[i])
            true_name = graph_data["class_names"][cls]
            match = "OK" if result["predicted_class"] == true_name else "MISS"
            logger.info(
                "[%s] True: %-20s | Pred: %-20s | conf=%.2f | %s",
                match,
                true_name,
                result["predicted_class"],
                result["confidence"],
                result["semantic_context"],
            )
            shown.add(cls)
        if len(shown) == graph_data["n_classes"]:
            break
    plot_training_curves()


# ══════════════════════════════════════════════════════════════
# РЕЖИМ 3: ТОЛЬКО ИНФЕРЕНС
# ══════════════════════════════════════════════════════════════

def run_inference_examples(graph_data: dict = None,
                            device: torch.device = None) -> None:
    if graph_data is None:
        graph_data = get_or_generate_graph_data()
    if device is None:
        device = get_device()

    logger.info("--- Inference Examples ---")

    try:
        analyzer = ThreatAnalyzer.from_checkpoint(
            checkpoint_path = PATHS["best_model"],
            graph_data      = graph_data,
            device          = device,
        )
    except FileNotFoundError:
        logger.warning("No checkpoint found - run training first")
        return

    # Берём по одному примеру каждого класса
    test_X = graph_data["incident_X"]
    test_y = graph_data["incident_y"]

    shown = set()
    for i in range(len(test_X)):
        cls = test_y[i].item()
        if cls not in shown:
            result = analyzer.analyze(test_X[i])
            true_name = graph_data["class_names"][cls]
            match = "OK  " if result["predicted_class"] == true_name else "MISS"
            logger.info(
                "[%s] True: %-20s | Pred: %-20s | conf=%.2f | sev=%-8s | ctx=%s",
                match,
                true_name,
                result["predicted_class"],
                result["confidence"],
                result["severity"],
                result["semantic_context"],
            )
            shown.add(cls)
        if len(shown) == graph_data["n_classes"]:
            break

    # Развёрнутое объяснение для одного инцидента
    logger.info("")
    logger.info("=== Detailed explanation for first incident ===")
    analyzer.explain(test_X[0], verbose=True)


# ══════════════════════════════════════════════════════════════
# РЕЖИМ 4: ТОЛЬКО ДАННЫЕ (для отладки датасета)
# ══════════════════════════════════════════════════════════════

def run_data_inspection(args: argparse.Namespace) -> None:
    """
    Генерирует данные и выводит статистику без обучения.

    Вызов:
        python main.py --mode data

    Полезно для:
        - Проверки Knowledge Graph (количество рёбер, типы)
        - Анализа дисбаланса классов
        - Визуализации семантической матрицы
    """
    logger.info("--- Data Inspection Mode ---")

    graph_data = get_or_generate_graph_data(force=True)

    # Статистика графа
    n_nodes = graph_data["n_classes"]
    n_edges = graph_data["edge_index"].shape[1]
    logger.info("Knowledge Graph: %d nodes, %d edges", n_nodes, n_edges)

    # Статистика инцидентов
    y = graph_data["incident_y"]
    logger.info("Total incidents: %d", len(y))
    for cls_idx, cls_name in enumerate(graph_data["class_names"]):
        count = (y == cls_idx).sum().item()
        pct   = count / len(y) * 100
        bar   = "#" * int(pct / 2)
        logger.info("  %20s: %4d (%5.1f%%) %s", cls_name, count, pct, bar)

    # Семантическая матрица
    plot_semantic_heatmap(
        graph_data["sim_matrix"],
        graph_data["class_names"],
    )
    logger.info("Semantic heatmap saved -> %s", PATHS["plots_dir"])

    # Типы рёбер
    edge_type  = graph_data["edge_type"]
    edge_names = DATA_CFG.edge_types
    logger.info("Edge types distribution:")
    for t_idx, t_name in enumerate(edge_names):
        count = (edge_type == t_idx).sum().item()
        logger.info("  %15s: %d edges", t_name, count)


# ══════════════════════════════════════════════════════════════
# CLI — АРГУМЕНТЫ КОМАНДНОЙ СТРОКИ
# ══════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    """
    Парсит аргументы командной строки.

    Примеры запуска:
        python main.py --mode full                  # полный пайплайн
        python main.py --mode train                 # только обучение
        python main.py --mode train --resume        # продолжить обучение
        python main.py --mode infer                 # только инференс
        python main.py --mode data                  # инспекция данных
        python main.py --mode full --regenerate_data# перегенерировать данные
    """
    parser = argparse.ArgumentParser(
        description="CyberThreat Intelligence AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "train", "infer", "data"],
        help="Режим запуска: full | train | infer | data",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Продолжить обучение с последнего чекпоинта",
    )
    parser.add_argument(
        "--regenerate_data",
        action="store_true",
        help="Принудительно перегенерировать данные",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Путь к конкретному чекпоинту (по умолчанию: best_model.pt)",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
        help="Уровень логирования",
    )

    # ★ TODO: Добавь свои аргументы для переопределения конфига:
    # parser.add_argument("--lr",         type=float, default=None)
    # parser.add_argument("--batch_size", type=int,   default=None)
    # parser.add_argument("--epochs",     type=int,   default=None)
    #
    # И переопредели в TRAIN_CFG:
    # if args.lr:
    #     TRAIN_CFG.learning_rate = args.lr

    return parser.parse_args()


# ══════════════════════════════════════════════════════════════
# ТОЧКА ВХОДА
# ══════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    # Логирование — первым делом
    setup_logging(log_level=args.log_level)
    logger.info("CyberThreat AI | mode=%s | device=%s",
                args.mode,
                "cuda" if torch.cuda.is_available() else "cpu")

    # Валидация конфига — до любых вычислений
    validate_config()

    # ── Маршрутизация по режимам ──────────────────────────────
    try:
        if args.mode == "full":
            run_full_pipeline(args)

        elif args.mode == "train":
            run_training(args)

        elif args.mode == "infer":
            graph_data = get_or_generate_graph_data()
            run_inference_examples(graph_data)

        elif args.mode == "data":
            run_data_inspection(args)

        else:
            logger.error("Неизвестный режим: %s", args.mode)
            sys.exit(1)

    except NotImplementedError as e:
        # NotImplementedError = намеренная заглушка (TODO)
        logger.error("\n%s\n%s\n%s", "─" * 55, str(e), "─" * 55)
        logger.error("Это намеренная заглушка — реализуй указанный блок.")
        sys.exit(2)

    except KeyboardInterrupt:
        logger.info("Прервано пользователем (Ctrl+C)")
        sys.exit(0)

    except Exception as e:
        logger.exception("Неожиданная ошибка: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()


# ══════════════════════════════════════════════════════════════
# ПОРЯДОК СБОРКИ — ЧЕКЛИСТ ДЛЯ ML ENGINEER
# ══════════════════════════════════════════════════════════════
#
# Шаги в правильном порядке:
#
# [ ] 1. Запусти `python main.py --mode data`
#         → убедись что граф генерируется, смотри статистику классов
#         → открой plots/semantic_heatmap.png
#
# [ ] 2. Реализуй в train.py:
#         → build_optimizer()       (вариант A или B)
#         → build_scheduler()       (cosine рекомендуется)
#         → _validate_one_epoch()   (копируй _train_one_epoch, убери backward)
#
# [ ] 3. В run_training() / run_full_pipeline() раскомментируй:
#         → model = build_model().to(device)
#         → optimizer = build_optimizer(model)
#         → scheduler = build_scheduler(optimizer)
#         → trainer = Trainer(...) и trainer.fit()
#
# [ ] 4. Запусти `python main.py --mode train`
#         → следи за логами: loss должен падать
#         → если loss не падает: снизь LR в 10x
#         → если loss взрывается: включи grad_clip или снизь LR
#
# [ ] 5. Раскомментируй evaluate_on_test и plot_training_curves
#
# [ ] 6. Запусти `python main.py --mode full`
#         → полный пайплайн: данные → обучение → тест → инференс
#
# [ ] 7. Раскомментируй блок ThreatAnalyzer в run_inference_examples
#         → запусти `python main.py --mode infer`
#
# ТИПИЧНЫЕ ПРОБЛЕМЫ И РЕШЕНИЯ:
#
# Loss не падает:
#     → learning_rate слишком мал → увеличь в 10x
#     → learning_rate слишком велик → уменьши в 10x
#     → проверь что optimizer.zero_grad() вызывается
#
# CUDA out of memory:
#     → уменьши batch_size в config.py
#     → включи AMP (use_amp=True) в config.py
#     → уменьши hidden_dim или num_heads
#
# Accuracy не растёт выше 60%:
#     → проверь label_smoothing (попробуй 0.05 вместо 0.10)
#     → увеличь num_gat_layers до 4
#     → добавь warmup LR (см. TODO в train.py)
#     → используй class_weights=True (уже включено)
#
# Early stopping срабатывает слишком рано:
#     → увеличь early_stopping_patience в config.py
#     → уменьши early_stopping_delta
#
# ══════════════════════════════════════════════════════════════